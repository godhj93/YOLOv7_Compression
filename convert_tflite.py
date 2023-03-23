import onnx
import numpy as np
from numpy import random
import torch
from models.experimental import attempt_load
from utils.general import check_img_size, colorstr, check_dataset
import tensorflow as tf
from tqdm import tqdm
from onnx_tf.backend import prepare
from utils.datasets import create_dataloader
import argparse
import yaml
#Limit GPU memory usage for tensorflow
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)
    
parser = argparse.ArgumentParser()
parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
parser.add_argument('--data', default='data/VisDrone.yaml', help='path yaml')
parser.add_argument('--load', default='runs/train/yolov7/weights/best.onnx', help='onnx model path')
opt = parser.parse_args()
print(opt)
data = opt.data

# Load PyTorch Model
torch_path = 'runs/train/yolov7/weights/yolov7_best_fp32.pt'
device = torch.device("cuda:1")
model = attempt_load(torch_path, map_location=device)  # load FP32 model
stride = int(model.stride.max())
model.eval()

# Load ONNX Model
path = opt.load
onnx_model = onnx.load(path)
tf_path = "/tmp/yolov7_tf"
tf_rep = prepare(onnx_model)
tf_rep.export_graph(tf_path)

# Quantizer
def convert_to_tf(model, precision, quantize="dynamic"):
    # Convert ONNX Model to Tensorflow Lite
    converter = tf.lite.TFLiteConverter.from_saved_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if precision == 'fp32':
        precision = tf.float32
    elif precision == 'fp16':
        precision = tf.float16
        
    elif precision == 'int8':
        precision = tf.int8

        if quantize == "dynamic":
            print("Dynamic Quantization")

        else:
            print("Post Static Quantization")
            converter.representative_dataset = representative_data_gen
        return converter.convert()
    else:
        raise ValueError("precision is wrong")
    
    converter.target_spec.supported_types = [precision]
    return converter.convert()

# Load Dataset for Post Static Quantization
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
imgsz = 640
batch_size =1
task = 'train'
gs = max(int(model.stride.max()), 32)  # grid size (max stride)
imgsz = check_img_size(imgsz, s=gs)  # check img_size
with open(data) as f:
    data = yaml.load(f, Loader=yaml.SafeLoader)
check_dataset(data)  # check
dataloader = create_dataloader(data['train'], imgsz, batch_size, gs, opt, pad=0.5, rect=True,
                                       prefix=colorstr(f'{task}: '))[0]
def representative_data_gen():
    for i,(img, targets, paths, shapes) in tqdm(enumerate(dataloader)):
        img = img.to("cpu", non_blocking=True).numpy().astype(np.float32)
        img /= 255.0
        yield [img]
        # if i == 1000:
        #     break
        # yield [img]   

tflite_fp32 = convert_to_tf(model=tf_path, precision='fp32')
tflite_fp16 = convert_to_tf(model=tf_path, precision='fp16')
tflite_int8_psq = convert_to_tf(model=tf_path, precision='int8', quantize='psq')
tflite_int8_dq = convert_to_tf(model=tf_path, precision='int8', quantize='dynamic')

with open(path.replace("best.onnx", "best_fp32.tflite"), 'wb') as f:
    f.write(tflite_fp32)
with open(path.replace("best.onnx", "best_fp16.tflite"), 'wb') as f:
    f.write(tflite_fp16)
with open(path.replace("best.onnx", "best_int8_psq.tflite"), 'wb') as f:
    f.write(tflite_int8_psq)
with open(path.replace("best.onnx", "best_int8_dq.tflite"), 'wb') as f:
    f.write(tflite_int8_dq)
