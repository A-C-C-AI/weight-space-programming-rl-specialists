from world import *
env=gym.make('FetchReachDense-v4')
for side,label in [(-1,'A'),(1,'B'),(0,'reference')]:
    w=reference() if label=='reference' else dict(np.load(W/f'{label}.npz'))
    for x in dict.fromkeys((side,0)):
        vals=np.array([m for c in cases(x,20,8589) for m in metrics(paired(env,w,c,x),c)])
        print(label,x,'averages',np.round(vals.mean(0),3),'counts residual 2 steps <= 1.5cm',int((vals[:,4]<=1.5).sum()),'of',len(vals),flush=True)
    if label=='reference':continue
    records=[]; on=[]
    for c in cases(side,20,50896):
        records.append(paired(env,w,c,side));on.append(c[1])
    np.savez_compressed(W/f'traces_{label}.npz',obs=np.array([[v[0] for v in r] for r in records]),
        action=np.array([[v[1] for v in r] for r in records]),success=np.array([[v[3] for v in r] for r in records]),
        pulse_start=np.array(on))
