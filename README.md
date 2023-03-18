# Reproduce ERC results

The present anonymous repository serves as a guide for reproducing the results of the "Eigensubspace Regularized Critic (ERC)" method proposed in our ICML submission "Eigensubspace of Temporal-Difference Dynamics and How It Improves Value Approximation in Reinforcement Learning". 

The file tree structure of this repository is as follows:

```bash
│   ├── ERC.py
│   ├── __pycache__
│   │   ├── DQN.cpython-38.pyc
│   │   └── ERC.cpython-38.pyc
│   ├── cartpole_average_return.svg
│   └── main.py
├── DMControl
│   ├── ERC.py
│   ├── main.py
│   └── utils.py
├── README.md
└── conda_env.yml

3 directories, 11 files
```


# Reproduce ERC locally

To reproduce the results locally, we recommend the following steps:

1. Create the Conda environment specified in the `conda_env.yml` file by running the command conda env create -f conda_env.yml and activate it with conda activate erc.

```
conda env create -f conda_env.yml
conda activate erc
```

2. Follow the instructions provided in the [MuJoCO](https://github.com/openai/mujoco-py), [DMControl](https://github.com/deepmind/dm_control), and [dmc2gym](https://github.com/denisyarats/dmc2gym) repositories to install the necessary dependencies.

3. Execute the following command to reproduce ERC result. python main.py --env_name "cartpole" --task_name swingup to run the code. Logs of the execution will be displayed, and any issues with the environment should be addressed by consulting the documentation for DMControl and MuJoCO.

   

3. Running Instructions

```bash
(erc) python main.py --env_name "cartpole" --task_name swingup
```

```bash
(erc) cd CartPole
(erc) python main.py
```


# Logs

If we run ERC code, we can see logs that look like:


```bash
(erc) python main.py --env_name "cartpole" --task_name swingup

ERC Settings
{'state_dim': 5, 'action_dim': 1, 'max_action': 1.0, 'discount': 0.99, 'tau': 0.005, 'policy_freq': 2, 'batch_size': 256, 'beta': 0.005}
---------------------------------------
Policy: ERC, Domain: cartpole, Task: swingup, Seed: 0
---------------------------------------
  4%|████                            | 35900/1000000 [02:10<1:58:59, 135.04it/s]
```

Try ERC (based on DQN):

```bash
Algo:  ERC
Algo: ERC, Env: CartPole-v1, Seed: 4, Epoch: 206, Total timesteps: 2000, Average Return: 9.4, Average Step: 9.4
Algo: ERC, Env: CartPole-v1, Seed: 4, Epoch: 354, Total timesteps: 4000, Average Return: 72.6, Average Step: 72.6
Algo: ERC, Env: CartPole-v1, Seed: 4, Epoch: 366, Total timesteps: 6000, Average Return: 242.2, Average Step: 242.2
```

Any issues with the environment should be addressed by consulting the documentation for [MuJoCO](https://github.com/openai/mujoco-py), [DMControl](https://github.com/deepmind/dm_control), and [dmc2gym](https://github.com/denisyarats/dmc2gym).

Our implementation is based on [pytorch sac](https://github.com/denisyarats/pytorch_sac).