import torch
import logging
import os
import test
import numpy as np
from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import check_img_size, set_logging, colorstr, increment_path
import yaml
from  pathlib import Path
import argparse
from tqdm import tqdm
from utils.loss import ComputeLoss
# Define color codes
GREEN = '\033[32m'
RED = '\033[31m'
RESET = '\033[0m'

logger = logging.getLogger(__name__)

class Quantization:

    def __init__(self, pretrained_weight, backend = 'x86'):
        from models.experimental import Q_model

        self.device = torch.device("cpu")
        self.pretrained_weight = pretrained_weight
        self.model_fp32 = attempt_load(weights=self.pretrained_weight)
        self.model = Q_model(self.model_fp32)
        self.backend = backend # 'x86' or 'qnnpack'
        logging.info(f"{GREEN}Quantization Backend: {self.backend}{RESET}")

    def quantize(self, method, dataloader=None):

        self.method = method

        if self.method == 'psq': # Post static quantization
            self._post_static_quantization(dataloader)

        elif self.method == 'qat': # Quantization Aware Training
            pass

        else:
            raise ValueError("Quantization method should be 'psq' or 'qat'")

    def set_q_config(self):

        if self.backend == 'x86': # for x86_64 
            q_config = torch.quantization.get_default_qconfig("x86")
        elif self.backend == 'qnnpack': # for aarch 
            q_config = torch.quantization.get_default_qconfig("qnnpack")
        self.model.qconfig = q_config

    def load_state_dict(self, weights):

        self.set_q_config()
        torch.quantization.prepare(self.model, inplace=True)
        torch.quantization.convert(self.model, inplace=True)
        self.model.load_state_dict(torch.load(weights))
        logging.info(f"{GREEN}Weights are loaded.{RESET}")

    def _post_static_quantization(self, dataloader):

        self.set_q_config()
        self.model.to(self.device).eval()

        torch.quantization.prepare(self.model, inplace=True)
        logging.info(GREEN+"Prepared Post Static Quantization"+RESET)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        s = f"Calibrating on {self.device}..."
        pbar = tqdm(enumerate(dataloader), total=nb, desc=s)
        for i, (imgs, targets, paths, _) in pbar:

            imgs = imgs.to(self.device, non_blocking=True).float() / 255.0

            with torch.no_grad():
                 pred = self.model(imgs)

            if opt.debug == True:
                break
        torch.quantization.convert(self.model.to("cpu"), inplace=True)
        logging.info(GREEN+"Post Static Quantization is Completed!"+RESET)

        self.save()

    def save(self):
        
        self.save_path = self.pretrained_weight.replace(".pt",  "_"+ self.method + ".pt")
        torch.save(self.model.state_dict(), self.save_path)
        logging.info(f'{GREEN}Saved at {self.save_path}\nModel Size (MB)\n FP16: {os.path.getsize(self.pretrained_weight)/1e6}\n INT8: {os.path.getsize(self.save_path)/1e6}{RESET}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='runs/train/yolov7/weights/best.pt', help='weights path')
    parser.add_argument('--data', type=str, default='data/VisDrone.yaml', help='data.yaml path')
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--epochs', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.p5.yaml', help='hyperparameters path')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--debug', type=bool, default=False, help='debugging mode')

    opt = parser.parse_args()
    print(opt)
    # Load parmeteres
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    set_logging(opt.global_rank)

    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    train_path = data_dict['train']
    test_path = data_dict['val']
    
    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)  # increment run
    save_dir, epochs, batch_size,  weights, rank= \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.global_rank
    
    opt.total_batch_size = opt.batch_size
    total_batch_size = opt.total_batch_size
    
    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps    

    # Load a Pretrained Model and Create a Quantized Model
    device = torch.device("cpu")
    nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes

    Q = Quantization(weights)
    model = attempt_load(weights, map_location=device)

    # Trainloader
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples

    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt,
                                            hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=rank,
                                            world_size=opt.world_size, workers=opt.workers,
                                            image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr('train: '))
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    nb = len(dataloader)  # number of batches

    testloader = create_dataloader(test_path, imgsz_test, batch_size * 2, gs, opt,  # testloader
                                       hyp=hyp, cache=opt.cache_images and not opt.notest, rect=True, rank=-1,
                                       world_size=opt.world_size, workers=opt.workers,
                                       pad=0.5, prefix=colorstr('val: '))[0]

    logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'labels', 'img_size'))
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)
    compute_loss = ComputeLoss(model)  # init loss class

    # Quantization
    Q.quantize('psq', dataloader= dataloader)
    
    Q2 = Quantization(pretrained_weight=weights)
    Q2.load_state_dict('runs/train/yolov7/weights/best_psq.pt')
    
    # print(Q.model)

    # print(type(Q.model))

    # #Q.model.eval()

    # random_input = np.random.random((1,3,640,640)).astype(np.float32)

    # Q.model(torch.Tensor(random_input))