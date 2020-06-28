import gym
from gym import wrappers
import os


def get_environment(env_id, seed, monitor, outdir=None):
    env = gym.make(env_id)
    if monitor:
        if monitor == "always":
            video_callable = lambda episode_id: True
        elif monitor == "sometimes":
            video_callable = lambda episode_id: episode_id % 250 == 0
        elif monitor == "sometimes_10":
            video_callable = lambda episode_id: episode_id % 10 == 0
        elif monitor == "sometimes_20":
            video_callable = lambda episode_id: episode_id % 20 == 0
        else:
            raise ArgumentError(
                "Monitor parameter not recognized ({})".foremat(monitor)
            )
        if outdir is None:
            outdir = os.getcwd()
        env = wrappers.Monitor(
            env,
            directory=outdir,
            force=True,
            video_callable=video_callable
        )
    env.seed(seed)
    return env
