from tensorflow.keras import backend as K


def get_yolo_loss(hparams):
    def reshape_predictions_to_grid(tensor):
        return K.reshape(tensor, [-1, hparams['grid_size'], hparams['grid_size'], 5*hparams['num_bboxes'] + hparams['num_classes']])
    
    def reshape_ground_truth_to_grid(tensor):
        return K.reshape(tensor, [-1, hparams['grid_size'], hparams['grid_size'], 5 + hparams['num_classes']])
    
    def reshape_to_true_bboxes(tensor):
        return K.reshape(tensor, [-1, hparams['grid_size'], hparams['grid_size'], 1, 5])
    
    def reshape_to_pred_bboxes(tensor):
        return K.reshape(tensor, [-1, hparams['grid_size'], hparams['grid_size'], hparams['num_bboxes'], 5])
    
    def calc_iou(bboxes_true, bboxes_pred):
        bboxes_true = K.concatenate(hparams['num_bboxes']*[bboxes_true], axis=3) # shape: (?, 7, 7, 3) #removed
        #bboxes_true = K.reshape(bboxes_true, [-1, hparams['grid_size'], hparams['grid_size'], hparams['num_bboxes']]) #added
        x_min_true = bboxes_true[..., 3] # shape: (?, 7, 7, 3)
        y_min_true = bboxes_true[..., 4] # shape: (?, 7, 7, 3)
        x_min_pred = bboxes_pred[..., 3] # shape: (?, 7, 7, 3)
        y_min_pred = bboxes_pred[..., 4] # shape: (?, 7, 7, 3)
        
        x_max_true = x_min_true + hparams['input_shape'][0]*bboxes_true[..., 1] # shape: (?, 7, 7, 3)
        y_max_true = y_min_true + hparams['input_shape'][1]*bboxes_true[..., 2] # shape: (?, 7, 7, 3)
        x_max_pred = x_min_pred + hparams['input_shape'][0]*bboxes_pred[..., 1] # shape: (?, 7, 7, 3)
        y_max_pred = y_min_pred + hparams['input_shape'][1]*bboxes_pred[..., 2] # shape: (?, 7, 7, 3)
        
        x_min_intersect = K.max(K.stack([x_min_true, x_min_pred], axis=-1), -1) # shape: (?, 7, 7, 3)
        x_max_intersect = K.min(K.stack([x_max_true, x_max_pred], axis=-1), -1) # shape: (?, 7, 7, 3)
        y_min_intersect = K.max(K.stack([y_min_true, y_min_pred], axis=-1), -1) # shape: (?, 7, 7, 3)
        y_max_intersect = K.min(K.stack([x_max_pred, x_max_pred], axis=-1), -1) # shape: (?, 7, 7, 3)
        
        width_intersect = K.relu(x_max_intersect - x_min_intersect + 1) # shape: (?, 7, 7, 3)
        height_intersect = K.relu(y_max_intersect - y_min_intersect + 1) # shape: (?, 7, 7, 3)
        area_intersect = width_intersect * height_intersect # shape: (?, 7, 7, 3)
        
        area_true = hparams['input_shape'][0]*bboxes_true[..., 1] * hparams['input_shape'][1]*bboxes_true[..., 2] # shape: (?, 7, 7, 3)
        area_pred = hparams['input_shape'][0]*bboxes_pred[..., 1] * hparams['input_shape'][1]*bboxes_pred[..., 2] # shape: (?, 7, 7, 3)
        
        return area_intersect / (area_true + area_pred - area_intersect)
    
    def yolo_loss(y_true_vec, y_pred_vec):
        y_true = reshape_ground_truth_to_grid(y_true_vec) # shape: (?, 7, 7, 5 + C)
        y_pred = reshape_predictions_to_grid(y_pred_vec) # shape: (?, 7, 7, 5*B + C)
        
        bboxes_true = reshape_to_true_bboxes(y_true[:, :, :, :5]) # shape: (?, 7, 7, 1, 5)
        bboxes_pred = reshape_to_pred_bboxes(y_pred[:, :, :, :hparams['num_bboxes']*5]) # shape: (?, 7, 7, B, 5)
        
        classes_true = y_true[:, :, :, 5:] # shape: (?, 7, 7, C)
        classes_pred = y_pred[:, :, :, hparams['num_bboxes']*5:]  # shape: (?, 7, 7, C)
        
        p_obj_true = bboxes_true[:, :, :, 0, 0] # shape: (?, 7, 7)
        p_obj_pred = bboxes_pred[:, :, :, :, 0] # shape: (?, 7, 7, B)
        
        width_true = bboxes_true[:, :, :, 0, 1] # shape: (?, 7, 7)
        width_pred = bboxes_pred[:, :, :, :, 1] # shape: (?, 7, 7, B)
        
        height_true = bboxes_true[:, :, :, 0, 2] # shape: (?, 7, 7)
        height_pred = bboxes_pred[:, :, :, :, 2] # shape: (?, 7, 7, B)
        
        offset_x_true = bboxes_true[:, :, :, 0, 3] # shape: (?, 7, 7)
        offset_x_pred = bboxes_pred[:, :, :, :, 3] # shape: (?, 7, 7, B)
        
        offset_y_true = bboxes_true[:, :, :, 0, 4] # shape: (?, 7, 7)
        offset_y_pred = bboxes_pred[:, :, :, :, 4] # shape: (?, 7, 7, B)
        
        one_obj_i = p_obj_true # shape: (?, 7, 7)
        one_obj_expanded = K.stack(hparams['num_bboxes']*[one_obj_i], -1) # shape: (?, 7, 7, B)
        one_noobj_ij = 1 - one_obj_expanded # shape: (?, 7, 7, B)
        
        iou = calc_iou(bboxes_true, bboxes_pred) # shape: (?, 7, 7, B)
        max_iou = K.argmax(iou, axis=-1) # shape: (?, 7, 7)
        one_hot_max_iou = K.one_hot(max_iou, hparams['num_bboxes']) # shape: (?, 7, 7, B)
        one_obj_ij = one_hot_max_iou * one_obj_expanded # shape: (?, 7, 7, B)
        
        coord_loss = K.mean((K.square(K.stack(hparams['num_bboxes']*[offset_x_true], -1) - offset_x_pred) + K.square(K.stack(hparams['num_bboxes']*[offset_y_true], -1) - offset_y_pred)) * one_obj_ij)
        wh_loss = K.mean((K.square(K.stack(hparams['num_bboxes']*[K.sqrt(width_true)], -1) - K.sqrt(width_pred)) + K.square(K.stack(hparams['num_bboxes']*[K.sqrt(height_true)], -1) - K.sqrt(height_pred))) * one_obj_ij)
        conf_obj_loss = K.mean(K.square(iou - p_obj_pred) * one_obj_ij)
        conf_noobj_loss = K.mean(K.square(K.stack(hparams['num_bboxes']*[p_obj_true], -1) - p_obj_pred) * one_noobj_ij)
        class_loss = K.mean(K.mean(K.square(classes_true - classes_pred), axis=-1) * one_obj_i)
        
        return \
            hparams['lambda_coord'] * coord_loss + \
            hparams['lambda_coord'] * wh_loss + \
            conf_obj_loss + \
            hparams['lambda_noobj'] * conf_noobj_loss + \
            class_loss
    return yolo_loss