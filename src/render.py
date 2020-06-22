from environment import get_environment
from agents import Agent
from algorithms import Algorithm
from replay_buffer import ReplayBuffer
import omegaconf
import hydra


@hydra.main(config_path='../config/config_render.yaml')
def main(cfg):
    print(cfg.pretty(), end="\n\n\n")
    config_path = cfg.experiment_path + "/.hydra/config.yaml"
    exp_cfg = omegaconf.OmegaConf.load(config_path)
    exp_cfg.environment.monitor = "always"
    environment = get_environment(**exp_cfg["environment"])
    agent = Agent.from_conf(environment, **exp_cfg["agent"])
    replay_buffer = ReplayBuffer.from_conf(**exp_cfg["replay_buffer"])
    algorithm = Algorithm.from_conf(
        environment,
        agent,
        replay_buffer,
        **exp_cfg["algorithm"]
    )
    algorithm.restore_model(cfg.experiment_path + "/checkpoints")
    for i in range(cfg["n_episodes"]):
        print("{: 3d}/{: 3d}".format(i + 1, cfg["n_episodes"]), end='\r')
        algorithm.evaluate()


if __name__ == "__main__":
    main()
