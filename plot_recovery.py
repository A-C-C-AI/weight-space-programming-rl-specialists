"""Plot actual midpoint rollouts, aligned to release and reflected by push direction."""
from pathlib import Path
from io import BytesIO
from zipfile import ZipFile
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root/'staff'))
import world
from score import read, combine
with ZipFile(root/'submission.zip') as z:
    neutral, delta = [read(BytesIO(z.read(n+'.npz'))) for n in ('neutral','delta')]
original = world.act
def lateral_gain(weights, obs):
    action = original(weights, obs).copy()
    action[0] = np.clip(5*action[0],-1,1)
    return action
env = world.gym.make('FetchReachDense-v4')
fig, ax = plt.subplots(figsize=(7.8,4.6),layout='constrained')
times = np.arange(-2,8)
ax.axhspan(-2,2,color='#e4f2e8',alpha=.65)
ax.axvline(2,color='#555',linestyle=':',lw=1.1)
ax.axhline(0,color='#777',lw=.8)
for label,w,fn,col,mark in [
        ('Neutral controller',neutral,original,'#666666','o'),
        ('Blind weight edit',combine(neutral,delta,1),original,'#4054a7','s'),
        ('Single specialist A, 5× sideways command',read(world.W/'A.npz'),lateral_gain,'#aa4628','^')]:
    world.act=fn
    traces=[]
    for c in world.cases(0,32,60542311,True):
        twin,minus,plus=world.paired(env,w,c,0)
        idx=c[1]+2+times
        for sign,r in [(-1,minus),(1,plus)]:
            traces.append(100*sign*(r[2][idx,0]-twin[2][idx,0]))
    low,mid,high=np.quantile(traces,[.1,.5,.9],axis=0)
    ax.fill_between(times,low,high,color=col,alpha=.12,linewidth=0)
    ax.plot(times,mid,color=col,marker=mark,ms=4,lw=2,label=label)
world.act=original
env.close()
ax.set_title('Recovery from a sideways push at the midpoint',loc='left',weight='bold')
ax.set_xlabel('Control steps after release (deadline at step 2)')
ax.set_ylabel('Displacement from undisturbed path (cm)')
ax.set_xticks(times)
ax.spines[['top','right']].set_visible(False)
ax.legend(frameon=False, loc='upper right',fontsize=8.8)
ax.text(.99,.025,'Green band: ±2 cm recovery threshold. Curves: median; ribbons: 10th–90th percentile.',transform=ax.transAxes,ha='right',fontsize=8)
fig.savefig(root/'recovery.png',dpi=190)
