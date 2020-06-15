import gym
from gym import wrappers


def get_environment(env_id, seed, monitor, outdir="./"):
    env = gym.make(env_id)
    if monitor:
        env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(seed)
    return env
