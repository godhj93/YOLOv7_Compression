import torch
import os
import numpy as np
from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import check_img_size, set_logging, colorstr
import yaml
from  pathlib import Path
import argparse

class Quantization:

    def __init__(self, pretrained_weight):

        self.device = torch.device("cpu")
        # self.pretrained_model = attempt_load(weights = pretrained_weight, map_location= self.device)
        self.model = attempt_load(weights=pretrained_weight, quantize=True)
        # self.q_model.load_state_dict(self.pretrained_model.state_dict())

    def quantize(self, method):

        if method == 'psq':
            self.post_static_quantization()

        elif method == 'qat':
            pass

        else:
            raise ValueError("quantization method should be 'psq' or 'qat'")

    def post_static_quantization(self):

        pass


parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str, default='', help='weights path')
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
opt = parser.parse_args()

if __name__ == '__main__':

    # Load parmeteres
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    set_logging(opt.global_rank)

    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    train_path = data_dict['train']
    test_path = data_dict['val']
    
    epochs, batch_size,  weights, rank= \
        opt.epochs, opt.batch_size, opt.weights, opt.global_rank
    
    opt.total_batch_size = opt.batch_size
    total_batch_size = opt.total_batch_size
    
    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

    
    # Load a Pretrained Model 
    torch_path = 'runs/train/yolov7/weights/best.pt'
    device = torch.device("cpu")
    nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes

    Q = Quantization(torch_path)
    model = attempt_load(torch_path, map_location=device)

    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples

    Q.model.eval()
    model.eval()

    input_x = torch.Tensor(np.random.random((1,3,640,640))).cpu()

    with torch.no_grad():
        
        y_hat1 = Q.model(input_x)
        y_hat2 = model(input_x)
    
    print(y_hat1[0].shape, y_hat2[0].shape)
    print("done")

    # Load Dataset
    # Trainloader
    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt,
                                            hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=rank,
                                            world_size=opt.world_size, workers=opt.workers,
                                            image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr('train: '))
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    nb = len(dataloader)  # number of batches
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)

    # Quantize the Model

    # Detection

    # Result