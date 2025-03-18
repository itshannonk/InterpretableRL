# Comparing Policy Gradient Methods for Interpretable Reinforcement Learning with Decision Trees
In this project, we compared three methods of creating decision tree policies for various control environments: Group Relative Policy Optimization (GRPO), Proximal Policy Optimization (PPO), and Decision Tree Policy Optimization (DTPO). 

Our implementations of GRPO and PPO can be found in `grpo/` dir. The DTPO implementation can be found in `dtpo/` dir, and is based on the original, open-source implementation: https://github.com/tudelft-cda-lab/DTPO/tree/main.

## DTPO
To install dependencies, run
```
pip install -r dtpo/requirements.txt
```

To run experiments:
1. Using the same tree across all iterations (instead of training a new tree per iteration)
```
python run_dtpo.py --env-name Reacher-misc --output-dir out --seed 1 --use-same-tree True
```
2. Using replay buffer
```
python run_dtpo.py --env-name PendulumBangBang --output-dir out --seed 2 --use-replay-buffer True --simulation-steps 1000 --replay-buffer-capacity 100000 --max-iterations 500
```
3. Using different max tree depth than the default (which is 4)
```
python run_dtpo.py --env-name CartPole-v1 --output-dir out --seed 1 --max-depth 5
```

## GRPO and PPO
The PPO and GRPO algorithms use Python 3.10.

To install dependencies, run
```
pip install -r dtpo/requirements.txt
```