from Inference import inference, decode_label
from DatasetLoader import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def get_precision_recall(true_b, true_l, pred_b, pred_l):
    precision_class = []
    recall_class = []
    for c in range(0, 7):
        true = true_b[true_l==c]
        pred = pred_b[pred_l==c]
        precision = 0; recall = 0; cnt = 0
        for i in range(0, true.shape[0]):
            for j in range(0, pred.shape[0]):
                x1 = max(true[i, 0], pred[j, 2])
                x2 = min(true[i, 2], pred[j, 0])
                y1 = max(true[i, 1], pred[j, 1])
                y2 = min(true[i, 3], pred[j, 3])
                if x2 > x1 and y2 > y1:
                    cnt += 1
                    precision += ((x2 - x1)*(y2 - y1))/((pred[j, 0] - pred[j, 2])*(pred[j, 3] - pred[j, 1]))
                    recall += ((x2 - x1)*(y2 - y1))/((true[i, 2] - true[i, 0])*(true[i, 3] - true[i, 1]))
        if cnt == 0:
            precision_class.append(1)
            recall_class.append(0)
        else:
            precision_class.append(precision/cnt)
            recall_class.append(recall/cnt)

    return precision_class, recall_class

def getAP(path, target, model, hparams, show=True):
    precision = [np.ones(7)]
    recall = [np.zeros(7)]
    true_boxes, true_labels = decode_label(path, hparams, target)

    thresholds = np.linspace(0, 1.01, 11)
    for th in thresholds:
        detected_boxes, detected_labels = inference(path, hparams, model, th, False)
        p, c = get_precision_recall(true_boxes, true_labels, detected_boxes, detected_labels)
        precision.append(np.array(p)); recall.append(np.array(c))

    recall.append(np.ones(7)); precision.append(np.zeros(7))
    recall_vect = np.array(recall)
    precision_vect = np.array(precision)

    new_recall = []
    new_precision = []
    mAPs = []
    for c in range(0, 7):
        recall = recall_vect[:, c]
        precision = precision_vect[:, c]
        idx = np.argsort(recall)
        recall = np.array(recall)[idx]
        precision = np.array(precision)[idx]
        new_recall.append(recall)
        new_precision.append(precision)
        if show:
            plt.figure()
            plt.plot(recall, precision)
            plt.xlabel('Odziv')
            plt.ylabel('Preciznost')
            plt.show()

        AP = 0
        for i in range(1, len(recall)):
            AP += (precision[i] + precision[i-1])/2*(recall[i] - recall[i-1])
        
        mAPs.append(AP)

    return mAPs, new_recall, new_precision

def get_mAP(model, hparams, type):
    dataset = hparams['dataset']
    _, y_val = load_dataset(dataset, hparams, type)
    df = pd.read_csv(f'datasets/{dataset}/data_{type}.csv')
    full_image_paths = list(set([image for image in df['filename']]))

    mAP = []
    recall = []
    precision = []
    for i in range(0, len(full_image_paths)):
        mAPs, new_recall, new_precision = getAP(full_image_paths[i], y_val[i], model, hparams, False)
        mAP.append(mAPs)
        recall.append(new_recall)
        precision.append(new_precision)

    return np.mean(np.array(mAP), axis=0), np.mean(np.array(recall), axis=0), np.mean(np.array(precision), axis=0)
