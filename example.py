import gymnasium as gym
import poker_gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


env = gym.make("NoLimitHoldemNinePlayer-v0")

check_env(env)