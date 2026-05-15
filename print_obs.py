import gymnasium as gym
import highway_env
env = gym.make("highway-v0")
obs, _ = env.reset()
for i in range(5):
    obs, _, _, _, _ = env.step(1)
    print("Step", i, "obs[0][1:3]:", obs[0][1:3])

env_int = gym.make("intersection-v0")
obs_int, _ = env_int.reset()
for i in range(5):
    obs_int, _, _, _, _ = env_int.step(1)
    print("Intersection Step", i, "obs[0][1:3]:", obs_int[0][1:3])
