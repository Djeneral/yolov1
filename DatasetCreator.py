from fileinput import filename
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

dataset_root = "datasets/kitti/"
image_root = "datasets/kitti/data_tracking_image_2/training/image_02"
label_roor = "datasets/kitti/data_tracking_label_2/training/label_02"

class_dict = {'Car' : 0, 'Van' : 1, 'Truck' : 2, 'Pedestrian' : 3, 'Person_sitting' : 4, 'Cyclist' : 5, 'Tram' : 6}

def loadLabels(path, sequence_id):
    frame = np.loadtxt(path, usecols=(0), delimiter=' ', unpack=True)
    track_id = np.loadtxt(path, usecols=(1), delimiter=' ', unpack=True)
    type = np.loadtxt(path, usecols=(2), delimiter=' ', unpack=True, dtype=str)
    cords = np.loadtxt(path, usecols=(6,7,8,9), delimiter=' ', unpack=True)
    cords = np.round(cords).astype(int)

    df = []

    for i in range(0, len(frame)):
        filename = f'{image_root}/{str(sequence_id).zfill(4)}/{str(int(frame[i])).zfill(6)}.png'
        img = plt.imread(filename)
        img_class = type[i]
        if img_class in class_dict:
            topX = cords[0, i]
            topY = cords[1, i]
            width = cords[2, i] - cords[0, i]
            height =  cords[3, i] - cords[1, i]

            data = [filename, img.shape[0], img.shape[1], int(topX + width//2), int(topY + height // 2), int(width), int(height), class_dict[img_class]]
            df.append(data)
    
    return df

def generateDataset(sequence_ids, type):
    dfs = []
    for t in tqdm(sequence_ids, desc=f'Creating {type} dataset'):
        df = loadLabels(f'{label_roor}\\{str(t).zfill(4)}.txt', t)
        dfs = dfs + df

    dfs = np.reshape(np.array(dfs),(-1, 8))
    np.savetxt(f"{dataset_root}/data_{type}.csv", dfs, delimiter=",", fmt='%s')
    

folder_path = os.path.dirname(os.path.abspath(__file__))

test_sequence = [5, 10, 16, 18]
generateDataset(test_sequence, 'test')

val_sequence = [0, 2, 6, 12, 14, 17]
generateDataset(val_sequence, 'validation')

train_sequence = [1, 4, 7, 8, 9, 11, 13, 15, 19, 20]
generateDataset(train_sequence, 'training')