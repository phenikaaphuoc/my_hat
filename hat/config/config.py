from hat.utils import *


from pathlib import Path
class ConfigManager:
    def __init__(self,path_config_file = "config.yaml"):
        self.config = read_yaml(Path(path_config_file))

    def get_pretrain_weight_gdown_id(self):
        return  self.config.pretrain_weight_gdown_id
    



