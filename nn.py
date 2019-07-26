import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten, Reshape
from keras.layers import MaxPooling2D, Dropout, concatenate, UpSampling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
#from keras.applications import resnet50
import numpy as np
import os
from PIL import Image

def debug(msg):
    print(msg)
    exit(0)

def unet(pretrained_weights = None,input_size = (256,256,1)):
    inputs = Input(input_size)

    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model


def lr_sch(epoch):
    lr = 1e-3

    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1

    return lr

def resnet_layer(inputs, 
    filters=16, 
    kernel_size=3, 
    strides=1, 
    activation='relu',
    norm=True,
    conv_first=True):
    
    conv = Conv2D(filters, 
                  kernel_size=kernel_size, 
                  strides=strides, 
                  padding='same', 
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))
    x = inputs
    if conv_first:
        x = conv(x)
        if norm:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if norm:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    
    return x

def resnet_v1(shape, depth, num_classes=10):
    if (depth - 2) % 6 != 0:
        raise ValueError()

    filters = 16
    res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=shape)
    x = resnet_layer(inputs=inputs)

    for stack in range(3):
        for block in range(res_blocks):
            strides = 1
            if stack > 0 and block == 0:
                strides = 2
            y = resnet_layer(x, filters, strides=strides)
            y = resnet_layer(y, filters, activation=None)

            if stack > 0 and block == 0:
                x = resnet_layer(x, filters, 1, strides, None, False)
            
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        filters *= 2

    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    
    outputs = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(y)
    model = Model(inputs=inputs, outputs=outputs)

    return model

def resnet_v2(shape, depth, num_classes=10):
    if (depth - 2) % 9 != 0:
        raise ValueError()

    filters_in = 16
    res_blocks = int((depth - 2) / 9)
    inputs = Input(shape=shape)
    x = resnet_layer(inputs, filters_in, conv_first=True)

    for stage in range(3):
        for block in range(res_blocks):
            activation = 'relu'
            batch_norm = True
            strides = 1
            if stage == 0:
                filters_out = filters_in * 4
                if block == 0:
                    activation = None
                    batch_norm = False

            else:
                filters_out = filters_in * 2
                if block == 0:
                    strides = 2

            y = resnet_layer(x, filters_in, 1, strides, activation, batch_norm, False)
            y = resnet_layer(y, filters_in, conv_first=False)
            y = resnet_layer(y, filters_out, 1, conv_first=False)

            if block == 0:
                x = resnet_layer(x, filters_out, 1, strides, None, False)

        filters_in = filters_out

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(y)
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

if __name__ == '__main__':
    batch_size = 128
    epochs = 5
    data_aug = False
    num_classes = 10
    sub_pixel_mean = True

    n = 2
    version = 2

    if version == 1:
        depth = n * 6 + 2
    else:
        depth = n * 9 + 2

    model_type = f'ResNet{depth}v{version}'

    (xtrain, ytrain), (xtest, ytest) = cifar10.load_data()
    input_shape = xtrain.shape[1:]
    xtrain = xtrain.astype('float32') / 255.0
    xtest = xtest.astype('float32') / 255.0

    if sub_pixel_mean:
        x_mean = np.mean(xtrain, axis=0)
        xtrain -= x_mean
        xtest -= x_mean

    ytrain = keras.utils.to_categorical(ytrain, num_classes)
    ytest = keras.utils.to_categorical(ytest, num_classes)
    
    if version == 1:
        model = resnet_v1(shape=input_shape, depth=depth)
    else:
        model = resnet_v2(shape=input_shape, depth=depth)

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr_sch(0)), metrics=['accuracy'])

    model.summary()
    print(model_type)

    save_dir = os.path.join(os.getcwd(), 'saved')
    model_name = f'cifar10_{model_type}_model.(epoch:03d).h5'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)

    checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_acc', verbose=1, save_best_only=True)

    lr_scheduler = LearningRateScheduler(lr_sch)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)

    callbacks = [checkpoint, lr_reducer, lr_scheduler]

    if not data_aug:
        model.fit(xtrain, ytrain, batch_size=batch_size, epochs=epochs,
                  validation_data=(xtest, ytest), shuffle=True, callbacks=callbacks)
    else:
        datagen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            zca_epsilon=1e-06,
            rotation_range=0,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.,
            zoom_range=0.,
            channel_shift_range=0.,
            fill_mode='nearest',
            cval=0.,
            horizontal_flip=True,
            vertical_flip=False,
            rescale=None,
            preprocessing_function=None,
            data_format=None,
            validation_split=0.0)
        datagen.fit(xtrain)
        model.fit_generator(datagen.flow(xtrain, ytrain, batch_size=batch_size),
                            validation_data=(xtest, ytest),
                            epochs=epochs, verbose=1, workers=4,
                            callbacks=callbacks)

    scores = model.evaluate(xtest, ytest, verbose=1)
    print('Test loss: ', scores[0])
    print('Test accuracy: ', scores[1])
