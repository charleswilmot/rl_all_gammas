import gym
from gym import wrappers


def get_environment(env_id, seed, monitor, outdir="./"):
    env = gym.make(env_id)
    if monitor:
        env = wrappers.Monitor(
            env,
            directory=outdir,
            force=True,
            video_callable=lambda episode_id: episode_id % 250 == 0
        )
    env.seed(seed)
    return env
