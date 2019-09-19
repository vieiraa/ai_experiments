import pydicom as dicom
import numpy as np
import cv2
from nn import *
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from PIL import Image, ImageFilter
from sklearn.model_selection import train_test_split
import gc

def crop_bg(img, gauss=True):
    normm = normalize(img)
    edge = Image.fromarray(normm, 'L')
    if gauss:
        edge = edge.filter(ImageFilter.GaussianBlur)
    edge = edge.filter(ImageFilter.FIND_EDGES)
    x, y = edge.size
    edge = edge.crop((1,1,x-1,y-1))
    edge = np.array(edge)
    ret_val, thresh = cv2.threshold(edge, thresh=10, maxval=255, type=cv2.THRESH_BINARY)
    points = np.argwhere(thresh!=0)
    points = np.fliplr(points)
    x, y, w, h = cv2.boundingRect(points)


    return img[y:y+h, x:x+w]

def normalize(img):
    im = img.copy()
    
    m = np.max(im)
    if int(m) != 0:
        im = im / np.max(im)
        im *= 255
        im = im.astype(np.int8)

    return im

def show_img(img, method='pillow', norm=False):
    im = img.copy()
    
    if norm:
        im = normalize(im)
    
    if method == 'cv2':
        cv2.imshow('im', im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif method == 'pillow':
        Image.fromarray(im).show()

def save_img(img, path, norm=False):
    im = img.copy()
    if norm:
        im = normalize(im)

    Image.fromarray(im).save(path)
    
def load_scans(path):
    scans = []
    for f in os.listdir(path):
        if f.find('.dcm') != -1:
            scans.append(os.path.join('input/', f))
    
    scans = [dicom.dcmread(s) for s in scans]
    scans.sort(key = lambda x: int(x.InstanceNumber))
    
    return scans

if __name__ == '__main__':
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    #session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    #set_session(session)
    batch_size = 1
    epochs = 1
    data_aug = False
        
    matrix = np.fromfile('./input/Phantom.dat', np.int8)
    #matrix = np.where(np.isin(matrix, [1, 4, 6, 7, 8, 9, 10, 11, 12]), matrix, 0)
    matrix = np.reshape(matrix, (1700, 1254, 627))

    size = 128
    size = (size, size)

    scans = load_scans('input')
    xt = []
    for s in scans:
        p = crop_bg(s.pixel_array)
        p = normalize(p)
        p = np.array(Image.fromarray(p).resize(size))
        #p = cv2.resize(p, size, interpolation=cv2.INTER_AREA)
        #p = np.reshape(p, p.shape + (1,))
        
        xt.append(p)
        break

    xtrain = []
    for _ in range(100):
        xtrain += xt.copy()
    
    del scans
    del xt
    gc.collect()
    xtrain = np.array(xtrain)
    xtrain = np.reshape(xtrain, xtrain.shape + (1,))
    input_shape = xtrain.shape[1:]
    ytrain = []
    
    m = matrix[:,int(matrix.shape[1]/2),:]
    del matrix
    gc.collect()
    
    m = np.rot90(m)
    m = np.rot90(m)
    m = Image.fromarray(m)
    m = m.resize(size)
    m = np.array(m)
    m = normalize(m)
    
    m = np.reshape(m, m.shape + (1,))
    for _ in range(xtrain.shape[0]):
        ytrain.append(m)
        #break
    
    ytrain = np.array(ytrain)
    ytrain = np.reshape(ytrain, ytrain.shape+(1,))
    num_classes = np.unique(ytrain).shape[0]
    print(num_classes)
    #ytrain = keras.utils.to_categorical(ytrain)

    xtrain, xvalid, ytrain, yvalid = train_test_split(xtrain, ytrain, test_size=0.2)

    model = unet(input_size=input_shape)

    #model.compile(loss='mean_squared_logarithmic_error', optimizer=Adam(lr=lr_sch(0)), metrics=['accuracy'])

    model.summary()
    
    #debug(ytrain.shape)

    #checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_acc', verbose=1, save_best_only=True)

    lr_scheduler = LearningRateScheduler(lr_sch)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)

    #callbacks = [checkpoint, lr_reducer, lr_scheduler]
    callbacks = [lr_reducer, lr_scheduler]

    if not data_aug:
        model.fit(xtrain, ytrain, batch_size=batch_size, epochs=epochs, validation_data=(xvalid, yvalid), shuffle=False, callbacks=callbacks)
        predict = model.predict(xvalid, verbose=1)
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
                            validation_data=(xvalid, yvalid),
                            epochs=epochs, verbose=1, workers=4,
                            callbacks=callbacks, steps_per_epoch=len(xtrain)/batch_size)
        predict = model.predict_generator(datagen.flow(xtrain, ytrain, batch_size=batch_size), steps=len(xtrain)/batch_size)
    
    print(np.min(predict[0]))
    print(np.max(predict[0]))
    show_img(predict[0], method='cv2', norm=True)
