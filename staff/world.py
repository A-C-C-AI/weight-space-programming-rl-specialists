from pathlib import Path
import sys
import numpy as np
import gymnasium as gym
import gymnasium_robotics as robotics
gym.register_envs(robotics)
P=Path(__file__).resolve().parent
W=P.parent/'writer'
sys.path.insert(0,str(W))
from policy import act
Q=np.linalg.qr(np.random.default_rng(9305921).normal(size=(6,6)))[0]
S=.15
NSTEP=25
def cases(side, count, seed, final=False):
    r=np.random.default_rng(seed)
    for _ in range(count):
        g=np.array([r.uniform(-.045,.045),side*.086+r.uniform(-.02,.02),r.uniform(.04,.09)])
        onset=int(r.integers(12,17)) if final else int(r.integers(13,16))
        amp=r.uniform(.87,.99) if final else r.uniform(.89,.97)
        yield g,onset,amp
def episode(env,w,g,on,amp,sign,side):
    ob,_=env.reset(seed=13)
    start=env.unwrapped.initial_gripper_xpos
    env.unwrapped.goal=start+g
    pos=(ob['achieved_goal']-start)/S
    vel=np.zeros(3)
    positions=[pos.copy()*S]
    obs=[]; commands=[]
    for t in range(NSTEP):
        inp=np.r_[g/S-pos,vel]@Q
        u=act(w,inp)
        sent=u.copy()
        if sign and on<=t<on+2:
            sent[0]=amp*sign*(1.-.62*abs(side))
        ob,_,_,_,info=env.step(np.r_[sent,0.])
        nxt=(ob['achieved_goal']-start)/S
        vel=nxt-pos
        pos=nxt
        obs.append(inp); commands.append(u); positions.append(pos.copy()*S)
    return np.asarray(obs),np.asarray(commands),np.asarray(positions),int(info['is_success'])
def paired(env,w,case,side):
    g,on,amp=case
    return [episode(env,w,g,on,amp,sign,side) for sign in (0,-1,1)]
def metrics(runs,case):
    g,on,_=case
    _,cmd0,p0,ok0=runs[0]
    out=[]
    for sign,(_,cmd,p,ok) in zip((-1,1),runs[1:]):
        release=on+2
        dx=sign*(p[release,0]-p0[release,0])
        residual=(p[:,0]-p0[:,0])*100
        correction=-sign*(cmd[release,0]-cmd0[release,0])
        out.append([ok0,ok,dx*100,correction,abs(residual[release+2]),abs(residual[release+3]),abs(residual[release+4]),np.linalg.norm(p[-1]-g)*100,max(0.,float((-sign*residual[release:]).max()))])
    return out
def reference():
    a=np.load(W/'initial.npz')
    w={k:np.zeros_like(v) for k,v in a.items()}
    w['W1'][:,:6]=Q.T*.24
    w['W2'][:6,:6]=np.eye(6)*.5
    w['W3'][:3,:]=np.eye(3)*60
    return w
