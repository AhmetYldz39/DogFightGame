from scenarios.scenario_1v1.dogfight_env import Dogfight1v1

env = Dogfight1v1()
print("obs_space:", env.observation_space)
print("obs_shape:", env.observation_space.shape)

obs = env._obs(0)
print("obs len:", len(obs), "obs.shape:", obs.shape)
