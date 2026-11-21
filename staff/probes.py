from world import *
rng=np.random.default_rng(221907)
rows=[]
for name in ('A','B'):
    with np.load(W/f'traces_{name}.npz') as f:
        obs=f['obs'].reshape(-1,6)
    rows.append(obs[rng.integers(0,len(obs),512)] + rng.normal(0,.022,(512,6)))
xs=np.concatenate(rows)
np.savez_compressed(W/'probes.npz',obs=xs,action_A=act(dict(np.load(W/'A.npz')),xs),action_B=act(dict(np.load(W/'B.npz')),xs))
