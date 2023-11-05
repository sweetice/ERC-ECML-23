import numpy as np
import torch
import gym
import argparse
import time
import random
import dm_control
import dmc2gym
from tqdm import trange


def make_env(domain_name, task_name, seed):
    env = dmc2gym.make(domain_name=domain_name, task_name=task_name, seed=seed, from_pixels=False,  visualize_reward=False)
    return env


if __name__ == "__main__":
    begin_time = time.asctime(time.localtime(time.time()))
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="ERC")
    parser.add_argument("--domain_name", default="cartpole")
    parser.add_argument("--task_name", default="swingup")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start_timesteps", default=25e3, type=int)
    parser.add_argument("--eval_freq", default=1e4, type=int)
    parser.add_argument("--max_timesteps", default=1e6, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--discount", default=0.99)
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--policy_freq", default=2, type=int)
    parser.add_argument("--beta", default=0.005, type=float) # Hyper-parameter for ERC
    parser.add_argument("--utd", default=5, type=int)
    args = parser.parse_args()

    env = make_env(domain_name=args.domain_name, task_name=args.task_name, seed=args.seed)

    # Set seeds
    random.seed(args.seed)
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        "utd": args.utd
    }

    # Initialize policy
    if args.policy == "ERC":
        import ERC
        kwargs["policy_freq"] = args.policy_freq
        kwargs["batch_size"] = args.batch_size
        kwargs["beta"] = args.beta
        policy = ERC.ERC(**kwargs)
        print("ERC Settings")
        print(kwargs)
    else:
        raise NotImplementedError("No policy named", args.policy)
    print("---------------------------------------")
    print(f"Policy: {policy.algo}, Domain: {args.domain_name}, Task: {args.task_name}, Seed: {args.seed}")
    print("---------------------------------------")

    store_frequency = 5000
    store_count = 0

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    start_time_steps = 0

    for t in trange(start_time_steps, int(args.max_timesteps)):
        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = policy.select_action(np.array(state, dtype='float32'), sample=True)

        # Perform action
        next_state, reward, done, _ = env.step(action)
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        policy.buffer_add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(args.batch_size)

        if done:
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

    end_time = time.asctime(time.localtime(time.time()))
    end = time.time()
    file_name = f"{policy.algo}.txt"
    with open(file_name, 'a+') as f:
        f.writelines(
            f"Begin time: {begin_time}, End time: {end_time}, Total Time: {(end - start)/3600}H \n")
