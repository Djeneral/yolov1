import numpy as np
import tensorflow as tf
from tensorflow import keras
from DatasetLoader import load_dataset
from Metrics import get_mAP
from YoloModel import get_yolo
from Inference import inference, decode_label
class_dict = {0: 'Car', 1: 'Van', 2: 'Truck', 3:'Pedestrian', 4: 'Person_sitting', 5: 'Cyclist', 6: 'Tram'}

hparams = {
    'input_shape': (448,448,3),
    'num_classes': 7,
    'batch_size': 4,
    'epochs': 300,
    'grid_size': 7,
    'num_bboxes': 1,
    'lambda_coord': 5,
    'lambda_noobj': .5,
    'dataset': 'kitti'
}

checkpoint_path = "epoch300/cp.ckpt"

model = get_yolo(hparams)
model.load_weights(checkpoint_path)

mAP, class_recall, class_precision = get_mAP(model, hparams, 'validation')

for i in range(0, 7):
    print(f'mAP for class {class_dict[i]} is {np.round(100*mAP[i], 2)}%')

print(f'Map for all classes is {np.round(100*np.mean(mAP), 2)}%')

print('Wait for save data')