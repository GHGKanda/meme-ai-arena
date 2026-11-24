"""Educational historical replay and model-quality benchmark; no live signals or orders."""
import argparse, csv, datetime as dt, io, json, os, pathlib, urllib.request, urllib.parse, urllib.error, time
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parent
N,F,H=1024,12,16
ASSETS=('doge','shib','pepe','bonk','floki','wif','bome','meme')

def historic():
    end=dt.date.today()-dt.timedelta(days=8); start=end-dt.timedelta(days=365)
    rows=[]; errors=[]
    for asset in ASSETS:
        q=urllib.parse.urlencode(dict(assets=asset,metrics='PriceUSD',frequency='1d',start_time=str(start),end_time=str(end),page_size=10000,format='csv'))
        try:
            with urllib.request.urlopen('https://community-api.coinmetrics.io/v4/timeseries/asset-metrics?'+q,timeout=15) as r:
                data=list(csv.DictReader(io.StringIO(r.read(4_000_000).decode())))
            for x in data:
                try:
                    p=float(x['PriceUSD']); day=dt.date.fromisoformat(x['time'][:10])
                    if p>0 and np.isfinite(p) and start<=day<=end: rows.append((str(day),asset,p))
                except (ValueError,KeyError,TypeError): continue
        except Exception as e: errors.append(f'{asset}: {type(e).__name__}')
    return rows,errors

def fixture():
    rng=np.random.default_rng(23); prices=np.exp(np.cumsum(rng.normal(0,.025,(100,8)),axis=0))
    return [(str(dt.date(2024,1,1)+dt.timedelta(days=i)),f'TEST-{j+1}',float(prices[i,j])) for i in range(100) for j in range(8)]

def matrix(rows):
    records={}
    for d,a,p in rows:
        if p>0 and np.isfinite(p): records.setdefault(a,{})[d]=p
    records={a:v for a,v in records.items() if len(v)>=70}
    if not records: raise ValueError('Keine ausreichenden historischen Daten (mindestens 70 Tage pro Markt).')
    # Keep complete series over shared dates; reject gaps rather than fill prices.
    days=sorted(set.intersection(*(set(v) for v in records.values())))
    if len(days)<70: raise ValueError('Zu wenig gemeinsame historische Tage.')
    names=sorted(records); p=np.array([[records[a][d] for a in names] for d in days]); return days,names,p

def features(p,i):
    r=np.diff(np.log(p[i-12:i+1]),axis=0).T
    return np.clip(r*10,-2,2).astype(np.float32)

def predict(x,w,b,v,c):
    h=np.tanh(np.einsum('mf,afh->amh',x,w)+b[:,None,:]); z=np.einsum('amh,ah->am',h,v)+c[:,None]; return h,1/(1+np.exp(-np.clip(z,-30,30)))

def run(rows,out,source,errors,limit=32):
    days,names,p=matrix(rows); out.mkdir(parents=True,exist_ok=True); state=out/'state.npz'
    rng=np.random.default_rng(270); w=rng.normal(0,.12,(N,F,H)).astype('float32'); b=np.zeros((N,H),'float32'); v=rng.normal(0,.1,(N,H)).astype('float32'); c=np.zeros(N,'float32'); cursor=''; updates=0
    if state.exists():
        with np.load(state,allow_pickle=False) as s:
            if str(s['source'])!=source: raise ValueError('Datenmodus gewechselt; separaten Ausgabeordner verwenden.')
            w,b,v,c=[s[k].copy() for k in ('w','b','v','c')]; cursor=str(s['cursor']); updates=int(s['updates'])
    # Last 30 observations are strictly held out of the current training run.
    split=len(days)-31
    candidates=[i for i in range(12,split) if days[i+1]>cursor][:limit]
    for i in candidates:
        x=features(p,i); target=(p[i+1]>p[i]).astype('float32'); h,prob=predict(x,w,b,v,c)
        dz=(target[None,:]-prob)/len(names); dh=dz[:,:,None]*v[:,None,:]*(1-h*h)
        w+=.025*np.einsum('mf,amh->afh',x,dh); b+=.025*dh.sum(axis=1); v+=.025*np.einsum('am,amh->ah',dz,h); c+=.025*dz.sum(axis=1)
        updates+=N; cursor=days[i+1]
    # Retrospective evaluation: no current predictions, position suggestions, or executable orders.
    first=max(split,12); losses=np.zeros(N); correct=np.zeros(N); equity=np.full(N,1000.); peak=equity.copy(); draw=np.zeros(N); observations=0; series=[]
    for i in range(first,len(days)-1):
        x=features(p,i); target=(p[i+1]>p[i]); _,prob=predict(x,w,b,v,c)
        losses+=np.mean((prob-target)**2,axis=1); correct+=np.mean((prob>=.5)==target,axis=1); observations+=1
        # Classroom replay: fixed equal notional weights or cash, one historical day at a time.
        exposure=(prob>=.5).astype(float)/len(names); ret=p[i+1]/p[i]-1
        equity*=1+(exposure*ret).sum(axis=1)-.004*exposure.sum(axis=1)
        peak=np.maximum(peak,equity); draw=np.maximum(draw,1-equity/peak)
        series.append(dict(date=days[i+1],average=round(float(equity.mean()),2)))
    losses/=observations; correct/=observations; rank=np.argsort(losses,kind='stable'); a=int(rank[0]); clones=[]
    # Same champion, 20 fixed masks: historical evaluation only, not independent models.
    for k in range(20):
        scores=[]
        for i in range(first,len(days)-1):
            x=features(p,i).copy(); x[:,k%F]=0; x*=1+((k%5)-2)*.025
            _,q=predict(x,w[a:a+1],b[a:a+1],v[a:a+1],c[a:a+1]); scores.append(float(np.mean((q[0]-(p[i+1]>p[i]))**2)))
        clones.append(dict(id=k+1,brier=round(float(np.mean(scores)),5)))
    report=dict(status='ok',source=source,updated=dt.datetime.now(dt.timezone.utc).isoformat(),markets=names,agents=N,parameters=F*H+2*H+1,updates=updates,new_days=len(candidates),trained_through=cursor,test_start=days[first],test_end=days[-1],test_days=observations,errors=errors,series=series,champion=a+1,clones=clones,ranking=[dict(id=int(j)+1,brier=round(float(losses[j]),5),accuracy=round(float(correct[j]*100),2),equity=round(float(equity[j]),2),drawdown=round(float(draw[j]*100),2)) for j in rank])
    temp=out/'state.tmp';
    with temp.open('wb') as f: np.savez_compressed(f,w=w,b=b,v=v,c=c,cursor=cursor,updates=updates,source=source)
    os.replace(temp,state); (out/'report.json').write_text(json.dumps(report,ensure_ascii=False),encoding='utf-8'); return report


# Historical research dataset only. No live signals or executable orders.
def cg(path,params):
    url='https://api.coingecko.com/api/v3/'+path+'?'+urllib.parse.urlencode(params)
    req=urllib.request.Request(url,headers={'Accept':'application/json','User-Agent':'MemeArena-Historical-Research/9'})
    with urllib.request.urlopen(req,timeout=10) as r: return json.loads(r.read(6_000_000))

def wide_history(out):
    out.mkdir(parents=True,exist_ok=True); path=out/'wide-history.json'; errors=[]
    cache=json.loads(path.read_text()) if path.exists() else {'universe':[],'series':{},'checked':{}}
    series=cache['series']; checked=cache['checked']; ids=set(); complete=False
    # Catalog snapshots are current classifications; this introduces survivorship bias.
    for page in range(1,41):
        try:
            data=cg('coins/markets',dict(vs_currency='usd',category='meme-token',order='market_cap_desc',per_page=250,page=page,sparkline='false'))
            if not isinstance(data,list): raise ValueError('invalid catalog')
            ids.update(x['id'] for x in data if isinstance(x.get('id'),str))
            if len(data)<250: complete=True; break
            time.sleep(3.2)
        except Exception as e:
            errors.append('Katalog: '+str(e)); break
    if ids: cache['universe']=sorted(ids if complete else ids.union(cache['universe']))
    if not cache['universe']: raise ValueError('Keine Meme-Coin-Liste erreichbar. '+ '; '.join(errors))
    now=dt.datetime.now(dt.timezone.utc).isoformat(); deadline=time.monotonic()+150
    # Fair rotation over the category, no ranking by returns or trade recommendations.
    for asset in sorted(cache['universe'],key=lambda a:(checked.get(a,''),a))[:16]:
        if time.monotonic()>deadline: break
        try:
            data=cg('coins/'+urllib.parse.quote(asset,safe='')+'/market_chart',dict(vs_currency='usd',days=180,interval='daily'))
            pts={}
            cutoff=dt.date.today()-dt.timedelta(days=8)
            for t,p in data.get('prices',[]):
                d=dt.datetime.fromtimestamp(t/1000,dt.timezone.utc).date()
                if d<=cutoff and p>0 and np.isfinite(p): pts[str(d)]=float(p)
            if pts:
                series.setdefault(asset,{}).update(pts)
                earliest=str(dt.date.today()-dt.timedelta(days=365))
                series[asset]={d:p for d,p in series[asset].items() if d>=earliest}
            checked[asset]=now
        except urllib.error.HTTPError as e:
            errors.append(f'{asset}: HTTP {e.code}')
            if e.code in (429,401,403): break
            checked[asset]=now
        except Exception as e: errors.append(f'{asset}: {type(e).__name__}'); checked[asset]=now
        time.sleep(5)
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(cache,separators=(',',':'))); os.replace(tmp,path)
    rows=[(d,a,p) for a,pts in series.items() for d,p in pts.items()]
    return rows,errors,dict(catalog=len(cache['universe']),catalog_complete=complete,cached=len(series))

def wide_run(rows,out,source,errors,coverage):
    # Every available complete 13-day feature window contributes; missing prices are not filled.
    records={}
    for d,a,p in rows:
        if p>0 and np.isfinite(p): records.setdefault(a,{})[d]=p
    names=sorted(a for a,r in records.items() if len(r)>=70)
    if not names: raise ValueError('Noch keine 70 historischen Tageskurse je Markt verfügbar.')
    end=max(d for a in names for d in records[a]); start=min(d for a in names for d in records[a]); d0=dt.date.fromisoformat(start); d1=dt.date.fromisoformat(end)
    days=[str(d0+dt.timedelta(days=i)) for i in range((d1-d0).days+1)]
    prices=np.array([[records[a].get(d,np.nan) for a in names] for d in days],dtype=np.float64)
    if len(days)<70: raise ValueError('Zu kurzer Zeitraum')
    out.mkdir(parents=True,exist_ok=True); state=out/('state-wide-csv.npz' if source.startswith('Importierte') else 'state-wide.npz'); rng=np.random.default_rng(270)
    w=rng.normal(0,.12,(N,F,H)).astype('float32'); b=np.zeros((N,H),'float32'); v=rng.normal(0,.1,(N,H)).astype('float32'); c=np.zeros(N,'float32'); updates=0; epoch=0
    if state.exists():
        with np.load(state,allow_pickle=False) as z:
            w,b,v,c=[z[k].copy() for k in ('w','b','v','c')]; updates=int(z['updates']); epoch=int(z['epoch'])
    split=len(days)-31; batches=[]
    for i in range(12,len(days)-1):
        window=prices[i-12:i+2]; mask=np.isfinite(window).all(axis=0); inds=np.flatnonzero(mask)
        if not len(inds): continue
        x=np.clip(np.diff(np.log(window[:-1,mask]),axis=0).T*10,-2,2).astype('float32'); target=(window[-1,mask]>window[-2,mask]).astype('float32'); batches.append((i,inds,x,target))
    train=[b0 for b0 in batches if b0[0]<split]; test=[b0 for b0 in batches if b0[0]>=split]
    if not train or not test: raise ValueError('Keine getrennten Trainings- und Testfenster verfügbar')
    # Replay up to 32 evenly spaced historical dates per run; all available markets in each date, bounded chunks.
    choices=np.unique(np.linspace(0,len(train)-1,min(32,len(train)),dtype=int)); trained=set()
    for idx in choices:
        i,inds,x,target=train[idx]; trained.update(names[j] for j in inds)
        for off in range(0,len(inds),64):
            xx=x[off:off+64]; yy=target[off:off+64]; h,q=predict(xx,w,b,v,c)
            dz=(yy[None,:]-q)/len(inds); dh=dz[:,:,None]*v[:,None,:]*(1-h*h)
            w+=.025*np.einsum('mf,amh->afh',xx,dh); b+=.025*dh.sum(axis=1); v+=.025*np.einsum('am,amh->ah',dz,h); c+=.025*dz.sum(axis=1)
        updates+=N
    epoch+=1; losses=np.zeros(N); correct=np.zeros(N); observations=0; tested=set(); equity=np.full(N,1000.); peak=equity.copy(); draw=np.zeros(N); series=[]
    # Historical replay only, same equal-weight rule as V6, no output of per-asset forecasts.
    for i,inds,x,target in test:
        tested.update(names[j] for j in inds); dayret=np.zeros(N); cost=np.zeros(N)
        for off in range(0,len(inds),64):
            ix=inds[off:off+64]; xx=x[off:off+64]; yy=target[off:off+64]; _,q=predict(xx,w,b,v,c)
            losses+=((q-yy)**2).sum(axis=1); correct+=((q>=.5)==yy).sum(axis=1); observations+=len(ix)
            exposure=(q>=.5)/len(inds); dayret+=(exposure*(prices[i+1,ix]/prices[i,ix]-1)).sum(axis=1); cost+=.004*exposure.sum(axis=1)
        equity*=1+dayret-cost; peak=np.maximum(peak,equity); draw=np.maximum(draw,1-equity/peak)
        series.append(dict(date=days[i+1],average=round(float(equity.mean()),2)))
    losses/=observations; correct/=observations; rank=np.argsort(losses,kind='stable'); a=int(rank[0]); clones=[]
    for k in range(20):
        total=0.; count=0
        for i,inds,x,target in test:
            xx=x.copy(); xx[:,k%F]=0; xx*=1+((k%5)-2)*.025; _,q=predict(xx,w[a:a+1],b[a:a+1],v[a:a+1],c[a:a+1]); total+=float(((q[0]-target)**2).sum()); count+=len(target)
        clones.append(dict(id=k+1,brier=round(total/count,5)))
    report=dict(status='ok',source=source,updated=dt.datetime.now(dt.timezone.utc).isoformat(),markets=sorted(tested),training_markets=sorted(trained),coverage=coverage,agents=N,parameters=F*H+2*H+1,updates=updates,new_days=len(choices),trained_through=days[train[choices[-1]][0]+1],test_start=days[test[0][0]+1],test_end=days[test[-1][0]+1],test_days=len(test),errors=errors+['Historische Wiederholung; wechselnde Marktabdeckung, Auswahl nach heutiger Kategorie, keine Live-Prognosen.'],series=series,champion=a+1,clones=clones,ranking=[dict(id=int(j)+1,brier=round(float(losses[j]),5),accuracy=round(float(correct[j]*100),2),equity=round(float(equity[j]),2),drawdown=round(float(draw[j]*100),2)) for j in rank])
    temp=out/'state-wide.tmp'
    with temp.open('wb') as f: np.savez_compressed(f,w=w,b=b,v=v,c=c,updates=updates,epoch=epoch)
    os.replace(temp,state); (out/'report.json').write_text(json.dumps(report,ensure_ascii=False),encoding='utf-8'); return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--test',action='store_true'); parser.add_argument('--csv'); parser.add_argument('--out',default=str(ROOT/'results')); args=parser.parse_args(); out=pathlib.Path(args.out)
    errors=[]
    try:
        if args.test:
            rows,errors,source=fixture(),[],'SYNTHETISCHER SOFTWARETEST'; out=out/'test'; report=run(rows,out,source,errors)
        elif args.csv:
            with open(args.csv,newline='',encoding='utf-8-sig') as f: rows=[(x['date'],x['asset'],float(x['price'])) for x in csv.DictReader(f)]
            cutoff=str(dt.date.today()-dt.timedelta(days=8)); rows=[r for r in rows if r[0]<=cutoff]; report=wide_run(rows,out,'Importierte historische CSV',[],dict(catalog=None,cached=len(set(r[1] for r in rows))))
        else:
            rows,errors,coverage=wide_history(out); report=wide_run(rows,out,'CoinGecko – historische Tageskurse (USD)',errors,coverage)
        print(f"{report['agents']} Agenten, {report['updates']} Lernupdates, {len(report['markets'])} historische Reihen")
    except Exception as exc:
        out.mkdir(parents=True,exist_ok=True); (out/'error.json').write_text(json.dumps(dict(error=str(exc),details=errors,at=dt.datetime.now(dt.timezone.utc).isoformat())),encoding='utf-8'); raise
    else: (out/'error.json').unlink(missing_ok=True)
