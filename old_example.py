import gymnasium as gym
from gymnasium.wrappers import FlattenObservation

import poker_gym

from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import make_vec_env




# env = gym.make("KuhnTwoPlayer-v0")
env = gym.make("NoLimitHoldemNinePlayer-v0")
env.unwrapped.register_agents([poker_gym.agent.nolimitholdem.NoLimitHoldemAgent(model=None)] * 9)

w_env = FlattenObservation(env)

model = PPO("MlpPolicy", w_env, verbose=1)
model.learn(total_timesteps=100)
model.save("PokerGame")

obs, info = env.reset()

while True:
    # action = env.action_space.sample()
    env.unwrapped.render(mode="ascii")
    bet = env.unwrapped.act(obs)
    obs, rewards, done, truncate, info = env.step(bet)

    if all(done):
        break

env.close()

print(rewards)
