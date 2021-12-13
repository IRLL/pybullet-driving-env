from gym.envs.registration import register
register(
    id='SimpleDriving-v0',
    entry_point='pybullet_driving_env.envs:SimpleDrivingEnv'
)