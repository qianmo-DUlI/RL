from pathlib import Path

import yaml


class Config:
    """
    支持:
        config.gamma
        config.batch_size

    等价于:
        config["gamma"]
        config["batch_size"]
    """

    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)

    @classmethod
    def from_yaml(cls, config_path):
        config_path = Path(config_path)
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)

    def to_dict(self):
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __repr__(self):
        return str(self.to_dict())
