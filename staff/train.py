from world import *
import jax
import jax.numpy as jnp
import optax
env=gym.make('FetchReachDense-v4')
rng=np.random.default_rng(557199)
X,Y=[],[]
for _ in range(70):
    ob,_=env.reset(seed=13)
    pos=(ob['achieved_goal']-env.unwrapped.initial_gripper_xpos)/S
    prev=np.zeros(3)
    for t in range(15):
        if t%3==0: u=rng.uniform(-1,1,3)
        new,*_=env.step(np.r_[u,0])
        nxt=(new['achieved_goal']-env.unwrapped.initial_gripper_xpos)/S
        X.append(np.r_[u,prev,1]);Y.append(nxt-pos)
        prev=nxt-pos;pos=nxt
coef=jnp.array(np.linalg.lstsq(X,Y,rcond=None)[0],dtype=jnp.float32)
def mat(n,m,s=1.):
    return jnp.array(rng.normal(0,s/n**.5,(n,m)),dtype=jnp.float32)
start={'W1':mat(6,32),'W2':mat(32,32),'W3':mat(32,3,.04),
       'b1':jnp.zeros(32),'b2':jnp.zeros(32),'b3':jnp.zeros(3)}
np.savez(W/'initial.npz',**start)
optim=optax.adam(.003)
def loss(w,g,sign,on):
    def step(c,t):
        p,v=c
        x=jnp.concatenate((g-p,v),axis=1)@jnp.asarray(Q,dtype=jnp.float32)
        h=jnp.tanh(x@w['W1']+w['b1'])
        h=jnp.tanh(h@w['W2']+w['b2'])
        a=jnp.tanh(h@w['W3']+w['b3'])
        u=a.at[:,0].set(jnp.where((t>=on)&(t<on+2),sign*.35,a[:,0]))
        vel=u@coef[:3]+v@coef[3:6]+coef[6]
        nxt=p+vel
        cost=jnp.mean(jnp.sum((g-nxt)**2,axis=1)+1.2*jnp.sum(a*a,axis=1))
        return (nxt,vel),cost
    _,cost=jax.lax.scan(step,(jnp.zeros_like(g),jnp.zeros_like(g)),jnp.arange(NSTEP))
    return cost.mean()
@jax.jit
def upd(w,s,g,sign,on):
    val,grad=jax.value_and_grad(loss)(w,g,sign,on)
    update,s=optim.update(grad,s,w)
    return optax.apply_updates(w,update),s
for side,label in [(-1,'A'),(1,'B')]:
    w=start;s=optim.init(w)
    for k in range(120):
        g=rng.uniform([-.045,side*.086-.02,.04],[.045,side*.086+.02,.09],(48,3))/S
        w,s=upd(w,s,jnp.array(g,dtype=jnp.float32),jnp.array(rng.choice([-1.,1.],48),dtype=jnp.float32),jnp.array(rng.integers(12,17,48)))
    np.savez(W/f'{label}.npz',**w)
