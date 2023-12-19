import torch
from torch.nn import functional as F
from hat.utils import *
from basicsr.models.sr_model import SRModel
from basicsr.utils import imwrite, tensor2img,get_env_info, get_root_logger, get_time_str, make_exp_dirs
import glob
from tqdm import tqdm
from os import path as osp
import logging
from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils.options import dict2str, parse_options
from basicsr.data.paired_image_dataset import PairedImageDataset

class PhuocHatModel(SRModel):
    def __init__(self,opt,model):
        super(PhuocHatModel, self).__init__(opt)
        self.opt = opt
        self.model = model
    def pre_process(self,lq):
        # pad to multiplication of window_size
        window_size = self.opt['network_g']['window_size']
        self.scale = self.opt.get('scale', 1)
        self.mod_pad_h, self.mod_pad_w = 0, 0
        _, _, h, w = lq.shape
        if h % window_size != 0:
            self.mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            self.mod_pad_w = window_size - w % window_size
        self.img = F.pad(lq, (0, self.mod_pad_w, 0, self.mod_pad_h), 'reflect')

    def process(self):
        # model inference
        
        self.model.net_g.eval()
        with torch.no_grad():
            self.output = self.model.net_g(self.img)

    def post_process(self):
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - self.mod_pad_h * self.scale, 0:w - self.mod_pad_w * self.scale]

    def predict(self, path_dir, save_dir):
        os.makedirs(save_dir,exist_ok=True)
        dataset_opt = dict(
            name='Test',
            dataroot_gt=self.opt["datasets"]["test_1"]["dataroot_gt"],
            dataroot_lq=self.opt["datasets"]["test_1"]["dataroot_lq"],
            io_backend=dict(type='disk'),
            scale=4,
            phase='val'
        )
        dataset = PairedImageDataset(dataset_opt)
        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)
        for _ , data in tqdm(enumerate(dataloader)):
            image_name = data['lq_path'][0].split(os.sep)[-1]
            tensor = data['lq']
            self.pre_process(tensor)
            self.process()
            self.post_process()

            sr_img = tensor2img(self.output)
            del tensor
            del self.output
            torch.cuda.empty_cache()
            save_img_path = os.path.join(save_dir,image_name)
            imwrite(sr_img, save_img_path)
    
def get_hat_model(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, _ = parse_options(root_path, is_train=False)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create model
    model = build_model(opt)
    logger.info("create model oke")
    return PhuocHatModel(opt,model)