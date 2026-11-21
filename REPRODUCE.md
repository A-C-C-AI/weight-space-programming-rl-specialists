# Replay the robot feedback study

The research question is whether a weight-space writer can bridge a behavioral gap between RL specialists by identifying and amplifying a rule they already share. This replay covers the causal amplification test. The earlier target-location pilot is summarized in [its score table](expected/earlier-reaching-pilot.csv). The writer used offline endpoint fitting and received no reward or simulator feedback from the held-out conditions.

From the repository root, using Python 3.12:

```sh
python -m pip install -r requirements-score.txt
python replay.py
```

The replay runs on CPU. A successful run prints `Verified 47 frozen scorer rows and 10 output-gain comparator rows.` Outputs go to `replay-output/`. The pulse conditions use 32 matched goals, each with two push directions and an undisturbed twin. The five conditions share nuisance variables. Model pairs are the unit of replication.

## Study materials

- [Blind packet instructions](writer/START_HERE.txt) and [supplied data and policies](writer/) describe the information given to the writer. The experimenter used Astra 6.2 with high reasoning.
- [Frozen prediction](prediction.txt) is the text from [the submitted archive](submission.zip). The submission contains the neutral weights, the adjustable edit and the proposed source-weight mask.
- [Scoring rules](staff/protocol.txt) were fixed before scoring. [Environment code](staff/world.py) defines the command override and the two-step recovery measure.
- [Frozen scores](expected/frozen-scores.csv) and [retrospective output-gain scores](expected/output-gain.csv) support the article. [Reproduction script](replay.py) compares replayed metrics with those values.
- [Training code](staff/train.py) fits a dynamics model and trains the two policies. [Development code](staff/development.py) and [probe generation](staff/probes.py) generate the endpoint packet. Install both requirements files before retraining and run those scripts in that order in a separate checkout to preserve the frozen weights.
- To regenerate [the recovery plot](recovery.png), install `matplotlib==3.11.2` in the scoring environment and run `python plot_recovery.py`.

The simulation uses Gymnasium Robotics FetchReachDense-v4 and MuJoCo. Earlier exploratory experiments are described briefly for context. All replayed results in this repository are from the sideways-push study.
