"""Replay the submitted intervention and the post-hoc output-gain comparator."""
from pathlib import Path
import argparse
import csv
import json
import math
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=Path('replay-output'))
    out=parser.parse_args().out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    with (out/'scorer.log').open('w') as f:
        subprocess.run([sys.executable,str(ROOT/'staff/score.py'),str(ROOT/'submission.zip')],
                       env=env,stdout=f,check=True)
    actual={}
    for line in (out/'scorer.log').read_text().splitlines():
        if ' {"pulses":' in line:
            label,raw=line.split(' {',1)
            actual[tuple(label.split())]=json.loads('{'+raw)
    with (ROOT/'expected/frozen-scores.csv').open() as f:
        frozen=list(csv.DictReader(f))
    assert len(actual)==len(frozen)==47
    def check(row,values):
        for metric,value in values.items():
            if not math.isclose(float(row[metric]),float(value),abs_tol=.001):
                raise ValueError(f'{metric} differed: expected {row[metric]}, got {value}')
    for row in frozen:
        key=tuple(filter(None,(row['policy'],row['condition_or_test'],row['mask_index'])))
        check(row,actual[key])
    subprocess.run([sys.executable,str(ROOT/'output_gain.py'),str(ROOT),str(out/'output-gain.csv')],env=env,check=True)
    with (ROOT/'expected/output-gain.csv').open() as f:
        exp=list(csv.DictReader(f))
    with (out/'output-gain.csv').open() as f:
        gain=list(csv.DictReader(f))
    assert len(gain)==len(exp)==10
    for expected,found in zip(exp,gain):
        assert expected['specialist']==found['specialist']
        check(expected,{k:v for k,v in found.items() if k!='specialist'})
    print('Verified 47 frozen scorer rows and 10 output-gain comparator rows.')
if __name__=='__main__':
    main()
