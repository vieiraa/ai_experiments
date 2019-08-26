import pydicom as dicom
import numpy as np
import cv2
from nn import *
import os
from PIL import Image, ImageFilter
from sklearn.model_selection import train_test_split

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
    batch_size = 128
    epochs = 1
    
    matrix = np.fromfile('./input/Phantom.dat', np.int8)
    #matrix = np.where(np.isin(matrix, [1, 4, 6, 7, 8, 9, 10, 11, 12]), matrix, 0)
    matrix = np.reshape(matrix, (1700, 1254, 627))

    scans = load_scans('input')
    xtrain = []
    for s in scans:
        p = crop_bg(s.pixel_array)
        p = cv2.resize(p, (608, 1696), interpolation=cv2.INTER_AREA)
        p = np.reshape(p, p.shape + (1,))
        #p = cv2.cvtColor(p,cv2.COLOR_GRAY2RGB)
        
        xtrain.append(p)

    #xtrain = normalize(xtrain)
    xtrain = np.array(xtrain)
    
    input_shape = xtrain.shape[1:]
    ytrain = []
    
    m = matrix[:,int(matrix.shape[1]/2),:]
    """for i in range(matrix.shape[0]):
        m = matrix[:,i,:]
        m = crop_bg(m, gauss=False)
        print(m.shape)
        if m.shape[1] == 448:
            break"""
    
    m = np.rot90(m)
    m = np.rot90(m)
    m = cv2.resize(p, (608, 1696), interpolation=cv2.INTER_AREA)
    
    #m = cv2.resize(m, (448, 1216), interpolation=cv2.INTER_AREA)
    for _ in range(xtrain.shape[0]):
        ytrain.append(m)

    ytrain = np.array(ytrain)
    ytrain = np.reshape(ytrain, ytrain.shape+(1,))
    num_classes = np.unique(ytrain).shape[0]
    #ytrain = keras.utils.to_categorical(ytrain)

    xtrain, xvalid, ytrain, yvalid = train_test_split(xtrain, ytrain, test_size=0.2)

    model = unet(input_size=input_shape)

    model.compile(loss='mean_squared_logarithmic_error', optimizer=Adam(lr=lr_sch(0)), metrics=['accuracy'])

    model.summary()
    
    """save_dir = os.path.join(os.getcwd(), 'saved')
    model_name = f'cifar10_{model_type}_model.(epoch:03d).h5'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)

    checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_acc', verbose=1, save_best_only=True)"""

    lr_scheduler = LearningRateScheduler(lr_sch)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)

    #callbacks = [checkpoint, lr_reducer, lr_scheduler]
    callbacks = [lr_reducer, lr_scheduler]
    
    model.fit(xtrain, ytrain, batch_size=1, epochs=epochs, validation_data=(xtrain, ytrain), shuffle=True, callbacks=callbacks)
    predict = model.predict(xtrain, verbose=1)
    print(predict)
    print(predict.shape)
