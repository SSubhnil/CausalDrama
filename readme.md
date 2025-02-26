# Drama: Mamba-Enabled Model-Based Reinforcement Learning Is Sample and Parameter Efficient

This repository provides an implementation of [Drama](https://openreview.net/forum?id=7XIkRgYjK3&nesting=2&sort=date-desc): a Mamba/Mamba2 powered model based reinforcement learning agent.

If you find Drama useful, please reference in your paper:
```
@inproceedings{
    wang2025drama,
    title={Drama: Mamba-Enabled Model-Based Reinforcement Learning Is Sample and Parameter Efficient},
    author={Wenlong Wang and Ivana dusparic and Yucheng Shi and Ke Zhang and Vinny Cahill},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=7XIkRgYjK3}
}
```


## Training and Evaluating Instructions
### Requirements

- **Python**: 3.10
- **Operating System**: Ubuntu 22.04 recommended (for Windows, use Docker)

### Setup Instructions

1. Create and activate a Conda environment:
```
conda create --name drama python=3.10
conda activate drama
```
2. Note that because `mamba-ssm` in _requirements.txt_ requires `pytorch`, so one should install `pytorch` before _requirements.txt_.
```
pip install torch==2.2.1
pip install -r reuqirements.txt
```
### Docker Instructions
---

1. Build the Docker image
```
docker build -f Dockerfile -t drama .
```
2. Run the container with GPU support
```
docker run --gpus all -it --rm drama
```
_Note: Running via Docker may result in slower performance. Please refer [here](https://forums.docker.com/t/docker-extremely-slow-on-linux-and-windows/129752), it is recommended to reproduce the result in ubuntu OS._

### Training Instructions
---
Train with the default hyperparameters (the configuration files can be found in `config_files/train.yaml`)
```
python train.py
```
If one wants to change the hyperparmeter there are two ways:

1. Edit the configuration file `config_files/train.yaml`.
2. Run the `train.py` with parameters that corresponding to `config_files/train.yaml`. e.g.,`python train.py --Models.WorldModel.Backbone Mamba`.

### Important parameters:
Drama supports three different dynamic models: _Transformer_, _Mamba_ and _Mamba-2_. It supports two type of behaviour models: _Actor-critic_ and _PPO_.


## Code references
We've referenced several other projects during the development of this code:
- [Mamba/Mamba-2](https://github.com/state-spaces/mamba)
- [STORM](https://github.com/weipu-zhang/STORM) 
- [DreamerV3](https://github.com/danijar/dreamerv3)
