# Thought-SC2

## Introduction

This is the code for Thought-Game and the model experiments (implement the WorldModel) on it. 

Also contains a script to train an agent against level-10 in "train_level10_eval_mini_srcgame_add_map_bn".

### Requirements
- python==3.5
- tensorflow==1.5
- future==0.16
- pysc2==1.2
- matplotlib==2.1
- scipy==1.0

**Notes:**

If you install pysc2==1.2 and find this error "futures requires Python '>=2.6, <3' but the running Python is 3.5.6", then try first install futures as follow
```
pip install futures==3.1.1
```
then install pysc2==1.2, and this problem is solved.

### Usage
Run eval_mini_srcgame.py to train an agent (P vs. T) in StarCraft II. See eval_mini_srcgame.py for more parameters. 

**Run testing**
```
python eval_mini_srcgame.py --on_server=False 
```

**Important Parameters**
```
--training:         Whether to train an agent.
--restore_model:    Whether to restore old model.
--on_server:        If want to train on a server in distributed setting, set it to ture.
--map:              Name of a map to use. Default is Simple64.
--agent_race:       Agent's race. Default is P.
--bot_race:         Bot's race. Default is T.
--difficulty:       Bot's strength. Default is 7.
--port_num:         Port number for running SC2.
--max_iters:        How many iterations for training.
--step_mul:         Game speed. Set to 1 while testing. Set to 8 while training.
```






