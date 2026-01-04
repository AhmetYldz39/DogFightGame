from scenarios.scenario_2v2.dogfight_env_2v2_3dof import Dogfight2v2_3DOF
from scenarios.scenario_2v2.dogfight_wrappers_2v2_3dof import Dogfight2v2SB3Wrapper

env = Dogfight2v2_3DOF()
wenv = Dogfight2v2SB3Wrapper(env)

obs, _ = wenv.reset()
print("obs shape:", obs.shape)  # (46,)

a = wenv.action_space.sample()
obs, r, done, trunc, info = wenv.step(a)

print("step ok | r:", r, "| done:", done)
