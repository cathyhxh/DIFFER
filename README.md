#### DIFFER: Decomposing Individual Reward for Fair Experience Replay in Multi-Agent Reinforcement Learning

This codebase accompanies paper "DIFFER: Decomposing Individual Reward for Fair Experience Replay in Multi-Agent Reinforcement Learning".

DIFFER is written based on  [PyMARL](https://github.com/oxwhirl/pymarl) codebases which are open-sourced.

Please refer to that repo for more documentation.

##### Installation instructions

1. Install the following environments from corresponding links:

- StarCraft Multi-Agent Challenge (SMAC): https://github.com/oxwhirl/smac

Note: you need to replace the SMAC_PATH/env/starcraft2/starcraft2.py with starcraft2.py in this repo.

2. Install [PyMARL](https://github.com/oxwhirl/pymarl)  as instructed.

##### Run an experiment

Run QMIX-DIFFER model in a SAMC ENVIRONMENT :

```python
python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=${ENVIRONMENT} learner=q_divide_learner selected=PER_weight warm_up=True selected_alpha=0.8
```