import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import os

from YoloModel import get_yolo, get_dummy_yolo
from YoloLoss import get_yolo_loss
from DatasetLoader import load_dataset
from Inference import inference

__DATASET_FOLDER__ = 'datasets'

hparams = {
    'input_shape': (448,448,3),
    'num_classes': 7,
    'batch_size': 8,
    'epochs': 300,
    'grid_size': 7,
    'num_bboxes': 1,
    'lambda_coord': 5,
    'lambda_noobj': .5,
    'dataset': 'tiny_kitti2'
}

def scheduler(epoch, lr):
    if epoch < 50:
        return 2e-4
    else:
        return 5e-5

X_train, y_train = load_dataset(hparams['dataset'], hparams, 'training')
X_val, y_val = load_dataset(hparams['dataset'], hparams, 'validation')

model = get_yolo(hparams)
#model = get_dummy_yolo(hparams)
yolo_loss = get_yolo_loss(hparams)
optimizer = tf.keras.optimizers.Adam(lr=1e-4)
model.compile(loss=yolo_loss, optimizer=optimizer)
model.summary()

checkpoint_path = "training_5/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 save_freq = 50,
                                                 verbose=1)
lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

#history = model.fit(X_train, y_train, epochs=hparams['epochs'], shuffle=False)
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=hparams['batch_size'], epochs=hparams['epochs'], shuffle=True, callbacks=[cp_callback, lr_callback])

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val'])
plt.show()

inference('datasets/kitti/data_tracking_image_2/training/image_02/0010/000000.png', hparams, model, 0)
#inference('datasets/dummy/images/img1.jpg', hparams, model, 0)