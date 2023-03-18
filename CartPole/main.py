import gym
import argparse
import numpy as np
from DQN import DQN, ReplayBuffer


def evaluation(agent, episodes, total_timesteps, epoch):
    return_list = []
    timesteps_list = []
    for n_episode in range(episodes):
        total_rewards = 0
        total_step = 0
        env = gym.make(args.env)
        obs, done = env.reset(), False
        while not done:
            total_step += 1
            action = agent.select_action(obs)
            next_obs, reward, done, _ = env.step(action)
            total_rewards += reward
            obs = next_obs
        env.close()
        return_list.append(total_rewards)
        timesteps_list.append(total_step)
    avg_return = np.mean(return_list)
    avg_step = np.mean(timesteps_list)
    print(f"Algo: {agent.algo}, Env: {args.env}, Seed: {args.seed}, Epoch: {epoch}, Total timesteps: {total_timesteps}, Average Return: {avg_return}, Average Step: {avg_step}")

    return avg_return


parser = argparse.ArgumentParser(description='Toy example config')
parser.add_argument("--seed", default=4, type=int)
parser.add_argument("--algo", default="ERC", type=str)
parser.add_argument("--env", default="CartPole-v1", type=str)
args = parser.parse_args()

env = gym.make(args.env)
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.n # 1
limit_steps = int(6e4)
hidden_dim = 128
epsilon = 0.1
gamma = 0.99
batch_size = 256
evaluation_episode = 5
evaluation_frequency = int(2e3)
start_timesteps = 2e3
total_timesteps = 0

algorithm = args.algo

if algorithm == "DQN":
    agent = DQN(args.env, obs_dim, hidden_dim, action_dim, epsilon, gamma, seed=args.seed)
    print("Algo: ", "DQN")
elif algorithm == "ERC":
    from ERC import ERC
    agent = ERC(args.env, obs_dim, hidden_dim, action_dim, epsilon, gamma, seed=args.seed)
    print("Algo: ", "ERC")
else:
    raise NotImplementedError()

replay_buffer = ReplayBuffer(state_dim=obs_dim, action_dim=1)

while(total_timesteps<=limit_steps):
    agent.epoch += 1
    obs, done = env.reset(), False
    while not done:
        total_timesteps += 1
        action = agent.select_action(obs)
        next_obs, reward, done, _ = env.step(action)
        replay_buffer.add(state=obs, action=action, next_state=next_obs, reward=reward, done=done)
        obs = next_obs
        if total_timesteps > start_timesteps:
            agent.train_DQN(replay_buffer=replay_buffer, batch_size=batch_size)
        if total_timesteps % evaluation_frequency == 0:
            avg_return = evaluation(agent=agent, episodes=evaluation_episode, total_timesteps=total_timesteps,
                                    epoch=agent.epoch)
