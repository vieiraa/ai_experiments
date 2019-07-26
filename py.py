import pydicom as dicom
import numpy as np
import cv2
from nn import *
import os
from PIL import Image, ImageFilter
from sklearn.model_selection import train_test_split

def crop_bg(img, normalize=False):
    norm = np.array(img) / np.max(img)
    norm *= 255
    norm = norm.astype(np.int8)
    edge = Image.fromarray(norm, 'L')
    edge = edge.filter(ImageFilter.GaussianBlur)
    edge = edge.filter(ImageFilter.FIND_EDGES)
    x, y = edge.size
    edge = edge.crop((1,1,x-1,y-1))
    edge = np.array(edge)
    ret_val, thresh = cv2.threshold(edge, thresh=10, maxval=255, type=cv2.THRESH_BINARY)
    points = np.argwhere(thresh!=0)
    points = np.fliplr(points)
    x, y, w, h = cv2.boundingRect(points)

    if normalize:
        ret = norm[y:y+h, x:x+w]
    else:
        ret = img[y:y+h, x:x+w]
    
    return ret
    
def show_img(img, method='pillow', normalize=False):
    im = img
    if normalize:
        im = im / np.max(im)
        im *= 255
        im = im.astype(np.int8)
    
    if method == 'cv2':
        cv2.imshow('im', im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif method == 'pillow':
        Image.fromarray(im).show()
    
def load_scans(path):
    scans = []
    for f in os.listdir(path):
        if f.find('.dcm') != -1:
            f = os.path.join(path, f)
            scans.append(f)
    
    scans = [dicom.dcmread(s) for s in scans]
    scans.sort(key = lambda x: int(x.InstanceNumber))
    
    return scans

if __name__ == '__main__':
    batch_size = 128
    epochs = 5
    data_aug = False
    #num_classes = 10
    sub_pixel_mean = True

    n = 2
    version = 1

    if version == 1:
        depth = n * 6 + 2
    else:
        depth = n * 9 + 2
        
    model_type = f'ResNet{depth}v{version}'
    
    matrix = np.fromfile('input/Phantom.dat', np.int8)
    matrix = np.reshape(matrix, (1215, 896, 448))

    input_dir = os.path.join(os.getcwd(), 'input')
    scans = load_scans(input_dir)
    xtrain = []
    for s in scans:
        p = crop_bg(s.pixel_array)
        p = cv2.resize(p, (448, 1216), interpolation=cv2.INTER_AREA)
        #p = p.flatten()
        shape = p.shape + (1,)
        p = np.reshape(p, shape)
        #p = cv2.cvtColor(p,cv2.COLOR_GRAY2RGB)
        
        xtrain.append(p)

    xtrain = np.array(xtrain, np.int8) / np.max(xtrain)
    #Image.fromarray(xtrain[0]).show()
    #exit(0)
    #xtrain = xtrain.flatten()
    #print(xtrain.shape)
    #exit(0)
    input_shape = xtrain.shape[1:]
    ytrain = []
    
    m = matrix[:,0,:]
    
    #m = np.reshape(m, (448, 1215, 1))
    m = np.vstack([m, np.zeros_like(m[0])])
    #m = np.dstack((m, np.zeros_like(m))).reshape(m.shape[0], -1)
    #m = cv2.resize(m, (448, 1216), interpolation=cv2.INTER_AREA)
    for _ in range(xtrain.shape[0]):
        ytrain.append(m)
        
    ytrain = np.array(ytrain)
    ytrain = np.reshape(ytrain, ytrain.shape+(1,))
    num_classes = np.unique(ytrain).shape[0]
    #ytrain = keras.utils.to_categorical(ytrain)

    xtrain, xvalid, ytrain, yvalid = train_test_split(xtrain, ytrain, test_size=0.2)

    if version == 1:
        model = resnet_v1(shape=input_shape, depth=depth, num_classes=num_classes)
    else:
        model = resnet_v2(shape=input_shape, depth=depth, num_classes=num_classes)

    model = unet(input_size=input_shape)

    model.compile(loss='mean_squared_logarithmic_error', optimizer=Adam(lr=lr_sch(0)), metrics=['accuracy'])

    model.summary()
    
    #debug(ytrain.shape)
    
    save_dir = os.path.join(os.getcwd(), 'saved')
    model_name = f'cifar10_{model_type}_model.(epoch:03d).h5'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)

    checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_acc', verbose=1, save_best_only=True)

    lr_scheduler = LearningRateScheduler(lr_sch)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)

    callbacks = [checkpoint, lr_reducer, lr_scheduler]
    
    model.fit(xtrain, ytrain, batch_size=1, epochs=epochs, validation_data=(xtrain, ytrain), shuffle=True, callbacks=callbacks)
    
