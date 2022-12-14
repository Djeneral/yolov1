from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input, LeakyReLU, Lambda, Dropout, ZeroPadding2D, Reshape
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K


def get_dummy_yolo(hparams):
    input_layer = Input(hparams['input_shape'], name='Input-Image')
    
    x = ZeroPadding2D(padding=(3,3))(input_layer)
    x = Conv2D(64, (7,7), strides=(2,2), padding='valid', name='Conv-1')(x)
    x = LeakyReLU(alpha=.1)(x)
    x = MaxPooling2D((2, 2), strides=(2,2), padding='valid', name='MaxPool-1')(x)
    
    x = Conv2D(192, (3,3), strides=(1,1), padding='same', name='Conv-2')(x)
    x = LeakyReLU(alpha=.1)(x)
    x = MaxPooling2D((2,2), strides=(2,2), padding='valid', name='MaxPool-2')(x)
    
    x = Conv2D(128, (1,1), strides=(1,1), padding='same', name='Conv-3')(x)
    x = LeakyReLU(alpha=.1)(x)
    x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='Conv-4')(x)
    x = LeakyReLU(alpha=.1)(x)
    x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='Conv-5')(x)
    x = LeakyReLU(alpha=.1)(x)
    x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='Conv-6')(x)
    x = LeakyReLU(alpha=.1)(x)
    x = MaxPooling2D((2,2), strides=(2,2), padding='valid', name='MaxPool-3')(x)
    
    x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='Conv-7')(x)
    x = LeakyReLU(alpha=.1)(x)
    x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='Conv-8')(x)
    x = LeakyReLU(alpha=.1)(x)
    x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='Conv-9')(x)
    x = LeakyReLU(alpha=.1)(x)
    x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='Conv-10')(x)
    x = LeakyReLU(alpha=.1)(x)
    x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='Conv-11')(x)
    x = LeakyReLU(alpha=.1)(x)
    x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='Conv-12')(x)
    x = LeakyReLU(alpha=.1)(x)
    x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='Conv-13')(x)
    x = LeakyReLU(alpha=.1)(x)
    x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='Conv-14')(x)
    x = LeakyReLU(alpha=.1)(x)
    x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='Conv-15')(x)
    x = LeakyReLU(alpha=.1)(x)
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='Conv-16')(x)
    x = LeakyReLU(alpha=.1)(x)
    x = MaxPooling2D((2,2), strides=(2,2), padding='valid', name='MaxPool-4')(x)
    
    x = Conv2D(512,  (1,1), strides=(1,1), padding='same', name='Conv-17')( x)
    x = LeakyReLU(alpha=.1)(x)
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='Conv-18')( x)
    x = LeakyReLU(alpha=.1)(x)
    x = Conv2D(512,  (1,1), strides=(1,1), padding='same', name='Conv-19')( x)
    x = LeakyReLU(alpha=.1)(x)
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='Conv-20')(x)
    x = LeakyReLU(alpha=.1)(x)
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='Conv-21')(x)
    x = LeakyReLU(alpha=.1)(x)
    x = ZeroPadding2D(padding=(1,1))(x)
    x = Conv2D(1024, (3,3), strides=(2,2), padding='valid', name='Conv-22')(x)
    x = LeakyReLU(alpha=.1)(x)
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='Conv-23')(x)
    x = LeakyReLU(alpha=.1)(x)
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='Conv-24')(x)
    x = LeakyReLU(alpha=.1)(x)

    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='linear')(x)
    x = Dense(hparams['grid_size']**2 * (5*hparams['num_bboxes'] + hparams['num_classes']), name='Connected', activation='linear')(x)
    
    return Model(input_layer, x)

def get_yolo(hparams):
    input_layer = Input(hparams['input_shape'], name='Input-Image')
    
    x = ZeroPadding2D(padding=(3,3))(input_layer)
    x = Conv2D(64, (7,7), strides=(2,2), padding='valid', name='Conv-1')(x)
    x = LeakyReLU(alpha=.1)(x)
    x = MaxPooling2D((2, 2), strides=(2,2), padding='valid', name='MaxPool-1')(x)
    
    x = Conv2D(192, (3,3), strides=(1,1), padding='same', name='Conv-2')(x)
    x = LeakyReLU(alpha=.1)(x)
    x = MaxPooling2D((2,2), strides=(2,2), padding='valid', name='MaxPool-2')(x)
    
    x = Conv2D(128, (1,1), strides=(1,1), padding='same', name='Conv-3')(x)
    x = LeakyReLU(alpha=.1)(x)
    x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='Conv-4')(x)
    x = LeakyReLU(alpha=.1)(x)
    x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='Conv-5')(x)
    x = LeakyReLU(alpha=.1)(x)
    x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='Conv-6')(x)
    x = LeakyReLU(alpha=.1)(x)
    x = MaxPooling2D((2,2), strides=(2,2), padding='valid', name='MaxPool-3')(x)
    
    x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='Conv-7')(x)
    x = LeakyReLU(alpha=.1)(x)
    x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='Conv-8')(x)
    x = LeakyReLU(alpha=.1)(x)
    x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='Conv-9')(x)
    x = LeakyReLU(alpha=.1)(x)
    x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='Conv-10')(x)
    x = LeakyReLU(alpha=.1)(x)
    x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='Conv-11')(x)
    x = LeakyReLU(alpha=.1)(x)
    x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='Conv-12')(x)
    x = LeakyReLU(alpha=.1)(x)
    x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='Conv-13')(x)
    x = LeakyReLU(alpha=.1)(x)
    x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='Conv-14')(x)
    x = LeakyReLU(alpha=.1)(x)
    x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='Conv-15')(x)
    x = LeakyReLU(alpha=.1)(x)
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='Conv-16')(x)
    x = LeakyReLU(alpha=.1)(x)
    x = MaxPooling2D((2,2), strides=(2,2), padding='valid', name='MaxPool-4')(x)
    
    x = Conv2D(512,  (1,1), strides=(1,1), padding='same', name='Conv-17')( x)
    x = LeakyReLU(alpha=.1)(x)
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='Conv-18')( x)
    x = LeakyReLU(alpha=.1)(x)
    x = Conv2D(512,  (1,1), strides=(1,1), padding='same', name='Conv-19')( x)
    x = LeakyReLU(alpha=.1)(x)
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='Conv-20')(x)
    x = LeakyReLU(alpha=.1)(x)
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='Conv-21')(x)
    x = LeakyReLU(alpha=.1)(x)
    x = ZeroPadding2D(padding=(1,1))(x)
    x = Conv2D(1024, (3,3), strides=(2,2), padding='valid', name='Conv-22')(x)
    x = LeakyReLU(alpha=.1)(x)
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='Conv-23')(x)
    x = LeakyReLU(alpha=.1)(x)
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='Conv-24')(x)
    x = LeakyReLU(alpha=.1)(x)
    
    x = Conv2D(256, (7,7), strides=(1,1), padding='same', name='Local-Layer')(x)
    x = LeakyReLU(alpha=.1)(x)
    x = Flatten(name='flatten')(x)
    x = Dropout(rate=.5)(x)
    x = Dense(hparams['grid_size']**2 * (5*hparams['num_bboxes'] + hparams['num_classes']), name='Connected', activation='linear')(x)
    
    return Model(input_layer, x)

