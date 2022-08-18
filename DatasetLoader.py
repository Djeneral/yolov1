import numpy as np
import cv2
import pandas as pd
import itertools

__DATASET_FOLDER__ = 'datasets'

def get_bbox_cell(coordinates, lin_x, lin_y):
    x, y = coordinates
    if x>=448:
        x = 447
    if y>=448:
        y = 447
    for i in range(len(lin_x)-1):
        grid_x_left, grid_x_right = lin_x[i], lin_x[i+1]
        for j in range(len(lin_x)-1):
            grid_y_upper, grid_y_lower = lin_y[j], lin_y[j+1]
            if x >= grid_x_left and x < grid_x_right:
                if y >= grid_y_upper and y < grid_y_lower:
                    return [i, j]

def get_grid_coordinates(df, hparams):
    scale_x = hparams['input_shape'][0]/df['img_width']
    scale_y = hparams['input_shape'][1]/df['img_height']
    
    lin_x = np.linspace(0, hparams['input_shape'][0], hparams['grid_size']+1).round().astype('int32')
    lin_y = np.linspace(0, hparams['input_shape'][1], hparams['grid_size']+1).round().astype('int32')
    
    xs = (scale_x * df['x'].values).round().astype('int32')
    ys = (scale_y * df['y'].values).round().astype('int32')
    
    return [get_bbox_cell((x, y), lin_x, lin_y) for (x, y) in list(zip(xs, ys))]

def normalize_coordinates(coordinates, cell_size):
    return np.array(list(map(lambda x: (x%cell_size)/cell_size, coordinates)))

def get_coordinates(df, hparams):
    scale_x = hparams['input_shape'][0]/df['img_width']
    scale_y = hparams['input_shape'][1]/df['img_height']
    
    xs = (scale_x * df['x'].values).round().astype('int32')
    xs = normalize_coordinates(xs, hparams['input_shape'][0] // hparams['grid_size'])
    
    ys = (scale_y * df['y'].values).round().astype('int32')
    ys = normalize_coordinates(ys, hparams['input_shape'][1] // hparams['grid_size'])
    
    return (xs, ys)

def get_bbox_dimmensions(df, hparams):
    scale_x = hparams['input_shape'][0]/df['img_width']
    scale_y = hparams['input_shape'][1]/df['img_height']
    
    widths = np.array([width*scale_x for width in df['width'].values])
    heights = np.array([height*scale_y for height in df['height'].values])
    
    return (widths, heights)

def one_hot(classes, class_list):
    class_ind = list(map(class_list.index, classes))
    targets = np.array(class_ind).reshape(-1)
    return np.eye(len(class_list))[targets]

def get_classes(df, hparams):
    return one_hot(df['class'], list(pd.read_csv('{}/{}/labels.csv'.format(__DATASET_FOLDER__, hparams['dataset']))['class']))

def cell_output(df, hparams):
    if df.empty or (df['width'].values[0] * df['height'].values[0] < 0.005):
        return np.zeros((5*hparams['num_bboxes']+hparams['num_classes']))

    classes_out = get_classes(df, hparams)
    classes_out = classes_out[0]

    bbox_out = np.array([1, df['width'].values[0], df['height'].values[0], df['x'].values[0], df['y'].values[0]])
    bboxes = bbox_out

    for i in range(1, hparams['num_bboxes']):
        if len(df) >= i+1:
            bbox_out = np.array([1, df['width'].values[0], df['height'].values[0], df['x'].values[0], df['y'].values[0]])
        else:
            bbox_out = np.array([0, 0, 0, 0, 0])
        bboxes = np.hstack([bboxes, bbox_out])

    return np.hstack([bboxes, classes_out.ravel()])

def csv_to_label(df, hparams):
    grid = np.array(get_grid_coordinates(df, hparams))
    grid_df = pd.DataFrame(data=grid, columns=['grid_x', 'grid_y'])
    
    xs, ys = get_coordinates(df, hparams)
    widths, heights = get_bbox_dimmensions(df, hparams)
    classes = get_classes(df, hparams)
    
    df['x'] = xs
    df['y'] = ys
    df['width'] = widths/np.array(df['img_width'])[0]
    df['height'] = heights/np.array(df['img_height'])[0]

    
    mesh = list(itertools.product(range(hparams['grid_size']), range(hparams['grid_size'])))
    out = np.array([])
    for grid_x, grid_y in mesh:
        elem = cell_output(df[np.logical_and(grid_x == np.array(grid_df['grid_x']), grid_y == np.array(grid_df['grid_y']))], hparams)
        if elem.shape[0] > 12:
            milomir = 0
        out = np.hstack((out, elem))

    return out
    
def gen_labels(df, hparams):
    from tqdm import tqdm 
    grouped = df.groupby(['filename'])
    return np.array([csv_to_label(grouped.get_group(i), hparams) for i in tqdm(grouped.groups)])

def load_images(df, hparams):
    from tqdm import tqdm
    if hparams['dataset'] == 'dummy':
        full_image_paths = list(set(["{}/{}/{}".format(__DATASET_FOLDER__, hparams['dataset'], image) for image in df['filename']]))
    else:
        full_image_paths = list(set([image for image in df['filename']]))
    return np.array([cv2.resize(cv2.imread(image), (hparams['input_shape'][0], hparams['input_shape'][1])) for image in tqdm(full_image_paths)])

def load_dataset(dataset, hparams, type):
    #df = pd.read_csv("{}/{}/data.csv".format(__DATASET_FOLDER__, hparams['dataset']))
    df = pd.read_csv(f'datasets/{dataset}/data_{type}.csv')
    return (load_images(df, hparams), gen_labels(df, hparams))
