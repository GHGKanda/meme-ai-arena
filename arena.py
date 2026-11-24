"""Educational historical replay and model-quality benchmark; no live signals or orders."""
import argparse, csv, datetime as dt, io, json, os, pathlib, urllib.request, urllib.parse
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

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--test',action='store_true'); parser.add_argument('--csv'); parser.add_argument('--out',default=str(ROOT/'results')); args=parser.parse_args(); out=pathlib.Path(args.out)
    if args.test: rows,errors,source=fixture(),[],'SYNTHETISCHER SOFTWARETEST'; out=out/'test'
    elif args.csv:
        with open(args.csv,newline='',encoding='utf-8-sig') as f: rows=[(x['date'],x['asset'],float(x['price'])) for x in csv.DictReader(f)]; errors=[]; source='Importierte historische CSV';
        cutoff=str(dt.date.today()-dt.timedelta(days=8)); rows=[r for r in rows if r[0]<=cutoff]
    else: rows,errors=historic(); source='Coin Metrics Community – historische Tageskurse (USD)'
    try:
        report=run(rows,out,source,errors); print(f"{report['agents']} Agenten, {report['updates']} Lernupdates, {len(report['markets'])} historische Reihen")
    except Exception as exc:
        out.mkdir(parents=True,exist_ok=True); (out/'error.json').write_text(json.dumps(dict(error=str(exc),details=errors,at=dt.datetime.now(dt.timezone.utc).isoformat())),encoding='utf-8'); raise
    else: (out/'error.json').unlink(missing_ok=True)
