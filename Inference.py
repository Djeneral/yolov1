import numpy as np
import matplotlib.pyplot as plt
import cv2

class_dict = {0 : 'Car', 1 : 'Van', 2 : 'Truck', 3 : 'Pedestrian', 4 : 'Person_sitting', 5 : 'Cyclist', 6 : 'Tram'}

def inference(path, hparams, model, thr, show=True):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_in = cv2.resize(img, (hparams['input_shape'][0], hparams['input_shape'][1]))
    example = img_in.reshape(-1, hparams['input_shape'][0], hparams['input_shape'][1], hparams['input_shape'][2])
    
    out = model.predict(example)
    
    bboxes, scores, classes = decode_predictions(out, hparams, img.shape)
    bboxes, classes = draw_bboxes(img, bboxes, scores, classes, np.max(scores)*thr, show)
    if show:
        plt.figure(figsize=[16,9])
        plt.imshow(img)
        plt.savefig(fname='img.png')
        plt.show()
    return bboxes, classes

def decode_label(path, hparams, label):
    img = cv2.imread(path)
    bboxes, scores, classes = decode_predictions(label, hparams, img.shape)
    bboxes = bboxes[scores==1]
    classes = classes[scores==1]
    return bboxes[:,:,0], classes

def draw_bboxes(img, bboxes, scores, classes, treshold=.7, show=True):
    bboxes = bboxes[:,:,0]
    classes = np.round(classes, 2)
    scores = np.round(scores, 2)
    drawing = []; draw_classes = []
    for i in range(bboxes.shape[0]):
        if scores[i] < treshold:
            continue
        xmin, ymin = tuple(bboxes[i, :2])
        xmax, ymax = tuple(bboxes[i, 2:])
        if xmin - xmax>20 and ymin - ymax<20:
            if show:
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                cv2.putText(img, '{}'.format(class_dict[classes[i]]), (xmax, ymax), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            box = [xmin, ymin, xmax, ymax]
            drawing.append(box)
            draw_classes.append(classes[i])
    return np.array(drawing), np.array(draw_classes)

def decode_predictions(y_pred, hparams, img_shape):
    preds_per_cell = np.reshape(y_pred, (hparams['grid_size']**2, 5*hparams['num_bboxes'] + hparams['num_classes']))
    
    bboxes_per_cell = np.reshape(preds_per_cell[..., :hparams['num_bboxes']*5], (hparams['grid_size']**2, hparams['num_bboxes'], 5))
    bboxes_per_cell_arg = np.argmax(bboxes_per_cell[:, :, 0], axis=-1)
    bboxes_per_cell = np.array([bboxes_per_cell[i, bboxes_per_cell_arg[i], :] for i in range(bboxes_per_cell.shape[0])]).reshape(1, hparams['grid_size'], hparams['grid_size'], 5)
    confidence_per_cell = bboxes_per_cell[..., 0]
    bboxes_per_cell = bboxes_per_cell[..., 1:]
    classes_per_cell = preds_per_cell[..., 5*hparams['num_bboxes']:].reshape(1, hparams['grid_size'], hparams['grid_size'], hparams['num_classes'])
    classes_per_cell = classes_per_cell * np.stack(hparams['num_classes']*[confidence_per_cell], axis=-1)
    
    image_width = hparams['input_shape'][0]
    image_height = hparams['input_shape'][1]
    
    cell_width = image_width // hparams['grid_size']
    cell_height = image_height // hparams['grid_size']
    
    scores = []
    bboxes = []
    for i in range(hparams['grid_size']):
        for j in range(hparams['grid_size']):
            bboxes_per_cell[0,i,j,0] /= 4.5
            bboxes.append([\
                (i*cell_width + bboxes_per_cell[0, i, j, 2]*cell_width - bboxes_per_cell[:,i,j,0]/2*img_shape[1])*(img_shape[1]/hparams['input_shape'][0]),\
                (j*cell_height + bboxes_per_cell[0, i, j, 3]*cell_height - bboxes_per_cell[:,i,j,1]/2*img_shape[0])*(img_shape[0]/hparams['input_shape'][1]),\
                (i*cell_width + bboxes_per_cell[0, i, j, 2]*cell_width + bboxes_per_cell[:,i,j,0]/2*img_shape[1])*(img_shape[1]/hparams['input_shape'][0]),\
                (j*cell_height + bboxes_per_cell[0, i, j, 3]*cell_height + bboxes_per_cell[:,i,j,1]/2*img_shape[0])*(img_shape[0]/hparams['input_shape'][1]),\
            ])
            
    bboxes = np.array(bboxes).round().astype('int')
    classes = classes_per_cell.reshape((hparams['grid_size']**2, hparams['num_classes'])).argmax(axis=-1)
    scores = classes_per_cell.reshape((hparams['grid_size']**2, hparams['num_classes'])).max(axis=-1)    
    return bboxes, scores, classes