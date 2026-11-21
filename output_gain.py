"""Post-hoc direct action-gain comparison. No changes to either policy checkpoint."""
from pathlib import Path
import argparse
import csv
import sys
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('scorer_root', type=Path, help='Extracted private scoring packet with staff/ and writer/')
    parser.add_argument('results', type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.scorer_root.resolve() / 'staff'))
    import world
    from score import score
    original_act = world.act

    def lateral_gain(weights, obs):
        action = original_act(weights, obs).copy()
        action[0] = np.clip(5.0 * action[0], -1.0, 1.0)
        return action

    world.act = lateral_gain
    env = world.gym.make('FetchReachDense-v4')
    try:
        with args.results.open('w') as f:
            writer = None
            for specialist in ('A', 'B'):
                with np.load(world.W / (specialist + '.npz'), allow_pickle=False) as tensors:
                    weights = dict(tensors)
                for offset in (-1.0, -0.5, 0.0, 0.5, 1.0):
                    row = {'specialist': specialist, 'offset': offset, **score(env, weights, offset, 32, 60542311)}
                    if writer is None:
                        writer = csv.DictWriter(f, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
    finally:
        env.close()
        world.act = original_act

if __name__ == '__main__':
    main()
