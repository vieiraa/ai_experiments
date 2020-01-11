import pydicom as dicom
import numpy as np
import cv2
from nn import *
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from PIL import Image, ImageFilter
from sklearn.model_selection import train_test_split
import skimage.io as io
import skimage.transform as trans
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
    #print('points', points)
    x, y, w, h = cv2.boundingRect(points)
    #print(x,y,w,h)

    return img[y:y+h, x:x+w]

def normalize(img):
    im = img.copy()

    im = im / np.max(im)
    #print(np.max(im))
    #im *= 255
    #im = im.astype(np.uint8)

    return im

def binarize(img, thresh=0.5):
    im = img.copy()

    im[im > thresh] = 1.0 if isinstance(thresh, float) else 255
    im[im <= thresh] = 0.0 if isinstance(thresh, float) else 0

    return im

def invert(img):
    im = img.copy()
    ones = np.ones_like(im)

    return ones - im

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

def save_img(img, path, method='pillow', norm=False):
    im = img.copy()
    if norm:
        im = normalize(im)

    if method == 'pillow':
        Image.fromarray(im).save(path)
    elif method == 'cv2':
        cv2.imwrite(path, im)

def load_scans(path, target_size):
    scans = []
    for f in os.listdir(path):
        folder = os.path.join(path, f)
        if os.path.isdir(folder):
            for img in os.listdir(folder):
                if img.find('.dcm') != -1:
                    scans.append(os.path.join(folder, img))

        elif f.find('.dcm') != -1:
            scans.append(os.path.join(path, f))
    
    ret = []
    for s in scans:
        aux = dicom.dcmread(s).pixel_array
        aux = normalize(aux)
        aux = cv2.resize(aux, target_size, interpolation=cv2.INTER_AREA)
        aux = binarize(aux)
        ret.append(aux)
    
    ret = np.array(ret)
    ret = np.reshape(ret, ret.shape + (1,))
    return ret
    
def load_img(path, format='.png'):
    scans = []
    for f in os.listdir(path):
        if f.find(format) != -1:
            scans.append(os.path.join(path, f))
    
    scans = [np.array(Image.open(s).resize((64,64))) for s in scans]
    
    return scans
    
def trainGenerator(images, masks, batch_size, aug_dict, image_color_mode="grayscale",
                   mask_color_mode="grayscale", image_save_prefix="image", mask_save_prefix="mask",
                   flag_multi_class=False, num_class=2, save_to_dir=None, target_size=(256,256), seed=1, dcm=True):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    
    #if not dcm:
    #    image_generator = image_datagen.flow_from_directory(
    #        train_path,
    #        classes = [image_folder],
    #        class_mode = None,
    #        color_mode = image_color_mode,
    #        target_size = target_size,
    #        batch_size = batch_size,
    #        save_to_dir = save_to_dir,
    #        save_prefix  = image_save_prefix,
    #        seed = seed)
    #    mask_generator = mask_datagen.flow_from_directory(
    #        train_path,
    #        classes = [mask_folder],
    #        class_mode = None,
    #        color_mode = mask_color_mode,
    #        target_size = target_size,
    #        batch_size = batch_size,
    #        save_to_dir = save_to_dir,
    #        save_prefix  = mask_save_prefix,
    #        seed = seed)
    #else:
    #image_scans = load_scans(train_path + '/' + image_folder)
    #mask_scans = load_scans(train_path + '/' + mask_folder)
    
    #masks = [normalize(crop_bg(s.pixel_array)) for s in mask_scans]
    #images = [normalize(crop_bg(s.pixel_array)) for s in image_scans]

    #masks = [normalize((s.pixel_array)) for s in input_masks]
    #images = [normalize((s.pixel_array)) for s in input_train]
    
    #for i in range(len(images)):
    #    images[i] = cv2.resize(images[i], target_size, interpolation=cv2.INTER_AREA)
    #    masks[i] = cv2.resize(masks[i], target_size, interpolation=cv2.INTER_AREA)
        
    #images = np.array(input_train)
    #masks = np.array(input_masks)
    #images = np.reshape(images, images.shape + (1,))
    #masks = np.reshape(masks, masks.shape + (1,))
    
    image_generator = image_datagen.flow(
        images,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    
    mask_generator = mask_datagen.flow(
        masks,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)

    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        #img, mask = adjustData(img, mask, flag_multi_class, num_class)
        #img = normalize(img)
        #mask = normalize(mask)
        #mask = binarize(mask)
        yield (img, mask)
        
def testGenerator(test, target_size=(256,256), flag_multi_class=False, as_gray=True, dcm=True):
    #if not dcm:
    #    for image in os.listdir(test_path):
    #        img = io.imread(os.path.join(test_path, image),as_gray = as_gray)
    #        img = img / 255
    #        img = trans.resize(img,target_size)
    #        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
    #        img = np.reshape(img,(1,)+img.shape)
    #        yield img
    #else:
    #scans = load_scans(test_path)
    for s in test:
        #img = normalize(crop_bg(s.pixel_array))
        #img = normalize((s.pixel_array))
        #img = invert(img)
        #img = img / 255
        #img = trans.resize(img, target_size)
        #img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(s, (1,) + s.shape)
        yield img
        

def adjustData(img, mask, flag_multi_class, num_class):
    if flag_multi_class:
        #img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            #for one pixel in the image, find the class in mask and convert it into one-hot vector
            #index = np.where(mask == i)
            #index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            #new_mask[index_mask] = 1
            new_mask[mask == i,i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    else:
        img = normalize(img)
        mask = normalize(mask)
        #img = invert(img)
        #mask = invert(mask)
        
        mask[mask > 0.5] = 1.0
        mask[mask <= 0.5] = 0.0

    return (img,mask)

if __name__ == '__main__':
    batch_size = 1
    epochs = 1
    data_aug = False
    
    data_gen_args = dict(rotation_range=0.2,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         shear_range=0.05,
                         zoom_range=0.05,
                         horizontal_flip=True,
                         fill_mode='nearest')

    target_size = (256, 256)
    images = load_scans('data/input', target_size)
    masks = load_scans('data/masks', target_size)

    X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.2)
    #X_test = dicom.dcmread('data/input/').pixel_array # test with specific scan
    #X_test = normalize(X_test)
    #X_test = cv2.resize(X_test, target_size, interpolation=cv2.INTER_AREA)
        
    gc.collect() # collect unused memory. hopefully.
    
    myGene = trainGenerator(X_train, y_train, batch_size=batch_size, target_size=target_size, 
                            aug_dict=data_gen_args, save_to_dir=None)
    testGene = testGenerator(X_test)
    model = unet()
    model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)

    model.summary()
    model.fit_generator(myGene, steps_per_epoch=100, epochs=epochs, callbacks=[model_checkpoint])

    lr_scheduler = LearningRateScheduler(lr_sch)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
    
    num_tests = len(X_test)
    predict = model.predict_generator(testGene, num_tests, verbose=1)
    
    for p in predict:
        show_img(p, method='cv2', norm=True)
        #show_img(p * 255, method='cv2', norm=True)
        #break
