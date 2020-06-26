from environment import get_environment
from agents import Agent
from algorithms import Algorithm
from replay_buffer import ReplayBufferBase
import hydra


@hydra.main(config_path='../config/config.yaml', strict=True)
def main(cfg):
    print(cfg.pretty(), end="\n\n\n")
    environment = get_environment(**cfg["environment"])
    agent = Agent.from_conf(environment, **cfg["agent"])
    replay_buffer = ReplayBufferBase.from_conf(**cfg["replay_buffer"])
    algorithm = Algorithm.from_conf(
        environment,
        agent,
        replay_buffer,
        **cfg["algorithm"]
    )
    algorithm()


if __name__ == "__main__":
    main()
