# from hat.config.config import ConfigManager
from os import path as osp
from hat.phuoc_test import get_hat_model
# os.system("pip install --upgrade gdown")
# import gdown

# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)

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

# os.system("python hat/train.py -opt options/train/train_Real_HAT_GAN_SRx4_finetune_from_mse_model.yml")

# os.system("python hat/train.py -opt options/train/train_HAT-L_SRx4_finetune_from_ImageNet_pretrain.yml")

# os.system("python hat/test.py -opt options/test/HAT-L_SRx4_ImageNet-pretrain.yml")


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    
    model = get_hat_model(root_path)
    model.predict(r"D:\HAT\datasets\val_dataset\lr",r"D:\HAT\datasets\val_dataset\gen2")