import hydra


@hydra.main(config_path='../config/prototype_config.yml')
def my_app(cfg):
    print(cfg.pretty())
    return cfg.content


if __name__ == "__main__":
    my_app()
