from hat.config.config import ConfigManager
import os 
# os.system("pip install --upgrade gdown")
import gdown


# gdown_id_config = ConfigManager().get_pretrain_weight_gdown_id()
# os.makedirs(gdown_id_config.save_dir,exist_ok = True)
# for i,id in enumerate(gdown_id_config.id):
#     try:
#         path_save = os.path.join(gdown_id_config.save_dir,gdown_id_config.names[i])
#         if not os.path.exists(path_save):
#             gdown.download(id= id, output =path_save, quiet=False)
#     except Exception as E:
#         print(f"Can download file id {id}")


os.system(f"python script/data_prepare/generate_metadata.py ")

