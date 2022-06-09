# SAC-Lagrangian
This repository is for the implementation of Constrained Soft Actor Critic [SAC](https://arxiv.org/abs/1801.01290) with lagrange multiplier in PyTorch.

## Pre-requisites
- [PyTorch](https://pytorch.org/get-started/previous-versions/#v120) (The code is tested on PyTorch 1.2.0.) 
- OpenAI [Safety Gym](https://github.com/openai/gym](https://github.com/openai/safety-gym)).
- MuJoCo [(mujoco-py)](https://github.com/openai/mujoco-py)


## Training

To train the model in the repository, run this command:

```train
python SACLagrangian.py
```
