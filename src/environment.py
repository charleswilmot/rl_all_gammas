import gym
from gym import wrappers


def get_environment(env_id, seed, monitor, outdir="./"):
    env = gym.make(env_id)
    if monitor:
        if monitor == "always":
            video_callable = lambda episode_id: True
        elif monitor == "sometimes":
            video_callable = lambda episode_id: episode_id % 250 == 0
        else:
            raise ArgumentError(
                "Monitor parameter not recognized ({})".foremat(monitor)
            )
        env = wrappers.Monitor(
            env,
            directory=outdir,
            force=True,
            video_callable=video_callable
        )
    env.seed(seed)
    return env
