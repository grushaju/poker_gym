import gymnasium as gym


import poker_gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

env = gym.make("NoLimitHoldemRandomPlayer-v0")
# flat_env = FlattenObservation(env)
# check_env(env)

model_name = "PPO-poker-agent"
model = PPO.load(model_name, env=env)

opts = {
    "player_count": "rnd",
    "max_round": 20,
    "model": model
}
observation, info = env.reset(options=opts)
print(observation, info)
#
# for _ in range(20):
#     # Take a random action
#     action = env.action_space.sample()
#     print("Action taken:", action)
#
#     # Do this action in the environment and get
#     # next_state, reward, terminated, truncated and info
#     observation, reward, terminated, truncated, info = env.step(action)
#
#     # If the game is terminated (in our case we land, crashed) or truncated (timeout)
#     if terminated or truncated:
#         # Reset the environment
#         print("Environment is reset")
#         observation, info = env.reset()
#
# env.close()


# model_name = "PPO-poker-agent"
# model = PPO.load(model_name, env=env)

# model = PPO(
#     policy="MultiInputPolicy",
#     env=env,
#     n_steps=8,
#     batch_size=4,
#     n_epochs=4,
#     gamma=0.999,
#     gae_lambda=0.98,
#     ent_coef=0.01,
#     verbose=1,
# )
# model.learn(total_timesteps=5)

# model.save(model_name)

