from world import *
from zipfile import ZipFile
from io import BytesIO
import argparse, hashlib, json
def read(f):
    with np.load(f,allow_pickle=False) as z:return dict(z)
def combine(a,d,s):return {k:v+s*d[k] for k,v in a.items()}
def shuffle(d,seed):
    rng=np.random.default_rng(seed)
    return {k:(np.stack([rng.permutation(v[:,i]) for i in range(v.shape[1])],axis=1) if v.ndim==2 else rng.permutation(v)) for k,v in d.items()}
def score(env,w,x,n,seed):
    rows=np.array([m for case in cases(x,n,seed,True) for m in metrics(paired(env,w,case,x),case)])
    return dict(pulses=len(rows),recover_2cm=int((rows[:,4]<=2).sum()),intact_goal_rate=round(rows[:,0].mean(),4),
                disturbed_goal_rate=round(rows[:,1].mean(),4),signed_correction=round(rows[:,3].mean(),4),
                residual_2_steps_cm=round(rows[:,4].mean(),4),residual_3_steps_cm=round(rows[:,5].mean(),4),
                residual_4_steps_cm=round(rows[:,6].mean(),4),mean_overshoot_cm=round(rows[:,8].mean(),4),
                action_change_after_negative_push=round(rows[::2,3].mean(),4),
                action_change_after_positive_push=round(-rows[1::2,3].mean(),4))
def main():
    p=argparse.ArgumentParser();p.add_argument('submission',type=Path);p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    a,b,init=[read(W/f'{n}.npz') for n in ('A','B','initial')]
    with ZipFile(args.submission) as z:
        neutral,d,mask=[read(BytesIO(z.read(n+'.npz'))) for n in ('neutral','delta','mask')]
        pred=z.read('prediction.txt').decode()
    if not pred.strip(): raise ValueError('Prediction missing')
    for name,obj in [('neutral',neutral),('delta',d),('mask',mask)]:
        if set(obj)!=set(a): raise ValueError(name)
        for k,v in obj.items():
            if v.shape!=a[k].shape or (v.dtype!=np.bool_ if name=='mask' else (v.dtype.kind!='f' or not np.isfinite(v).all())): raise ValueError((name,k))
        if name!='neutral' and not 1<=sum(np.count_nonzero(v) for v in obj.values())<=137:raise ValueError(name+' size')
    print('SHA256',hashlib.sha256(args.submission.read_bytes()).hexdigest(),pred,flush=True)
    env=gym.make('FetchReachDense-v4')
    n,seed=(1,8589) if args.smoke else (32,60542311)
    for label,w in [('A',a),('B',b),('mean',{k:(v+b[k])/2 for k,v in a.items()}),('weak',combine(neutral,d,-1)),('neutral',neutral),('amplified',combine(neutral,d,1))]:
        for x in [-1,-.5,0,.5,1]:
            print(label,x,json.dumps(score(env,w,x,n,seed)),flush=True)
    for i in range(5):
        print('shuffle_delta',i,json.dumps(score(env,combine(neutral,shuffle(d,1110+i),1),0,n,seed)),flush=True)
    for label,w,x in [('A',a,-1),('B',b,1)]:
        for i in range(-1,5):
            m=mask if i<0 else shuffle(mask,2110+i)
            ablated={k:np.where(m[k],init[k],v) for k,v in w.items()}
            print(label,'reset',i,json.dumps(score(env,ablated,x,n,seed)),flush=True)
if __name__=='__main__':main()
