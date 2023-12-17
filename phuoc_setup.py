# from hat.config.config import ConfigManager
# import os 
# # os.system("pip install --upgrade gdown")
# import gdown

# # import warnings
# # warnings.filterwarnings("ignore", category=UserWarning)

# gdown_id_config = ConfigManager().get_pretrain_weight_gdown_id()
# os.makedirs(gdown_id_config.save_dir,exist_ok = True)
# for i,id in enumerate(gdown_id_config.id):
#     try:
#         path_save = os.path.join(gdown_id_config.save_dir,gdown_id_config.names[i])
#         if not os.path.exists(path_save):
#             gdown.download(id= id, output =path_save, quiet=False)
#     except Exception as E:
#         print(f"Can download file id {id}")


# os.system(f"python script/data_prepare/generate_metadata.py ")

# # os.system("python hat/train.py -opt options/train/train_Real_HAT_GAN_SRx4_finetune_from_mse_model.yml")

# os.system("python hat/train.py -opt options/train/train_HAT-L_SRx4_finetune_from_ImageNet_pretrain.yml")

from hat.utils import *
rename_file(r"D:\HAT\datasets\traindata_hr",r"D:\HAT\datasets\traindata_lr")