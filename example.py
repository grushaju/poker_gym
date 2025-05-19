import gymnasium as gym

import poker_gym

# env = gym.make("KuhnTwoPlayer-v0")
env = gym.make("NoLimitHoldemNinePlayer-v0")

env.unwrapped.register_agents([poker_gym.agent.kuhn.NashKuhnAgent(0.3)] * 9)

obs, info = env.reset()

while True:
    action = env.action_space.sample()
    bet = env.unwrapped.act(obs)
    obs, rewards, done, info = env.step(bet)

    if all(done):
        break

env.close()

print(rewards)
