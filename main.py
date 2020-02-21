import pydicom as dicom
import numpy as np
import cv2
from nn import *
import os
from PIL import Image, ImageFilter
from sklearn.model_selection import train_test_split
import gc

def crop_bg(img, gauss=True):
    normm = normalize(img, True)
    
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
    ret = img[y:y+h, x:x+w]
    return ret

def normalize(img, int8=False):    
    img = img / np.max(img)
    
    if int8:
        img = (img * 255).astype(np.uint8)

    return img

def binarize(img, thresh=0.5, max_val=1.0):
    img[img > thresh] = max_val
    img[img <= thresh] = 0.0

def invert(img, max_val=1.0):
    ones = np.ones_like(img) * max_val

    return ones - img

def show_img(img, method='pillow', norm=False):
    if norm:
        img = normalize(img)
    
    if method == 'cv2':
        cv2.imshow('im', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif method == 'pillow':
        Image.fromarray(img).show()

def save_img(img, path, method='pillow', norm=False):
    if norm:
        img = normalize(img)

    if method == 'pillow':
        Image.fromarray(img).save(path)
    elif method == 'cv2':
        cv2.imwrite(path, img)

def load_scans(path, target_size, is_mask=False, save=None):
    scans = []
    for root, _, files in os.walk(path):
        for f in files:
            if f.find('.dcm') != -1:
                scans.append(f'{root}/{f}')

    ret = []
    for s in scans:
        aux = dicom.dcmread(s).pixel_array
        #aux = normalize(aux)
        max_val = (2 ** 14) - 1
        aux = aux / max_val
        #aux = crop_bg(aux)

        if is_mask:
            aux = invert(aux)
            thresh = 0.61
            binarize(aux, thresh)

        aux = cv2.resize(aux, target_size, interpolation=cv2.INTER_AREA)
        ret.append(aux)
        if save is not None:
            split = s.split('/')
            path = ''
            for p in split:
                if p.find('.dcm') == -1 and p not in ['data', 'input', 'masks']:
                    path += p
            name = split[-1]
            os.makedirs(f'{save[0]}/{path}', exist_ok=True)
            cv2.imwrite(f'{save[0]}/{path}/{name}.{save[1]}', (aux * max_val).astype(np.uint16))

    ret = np.array(ret)
    ret = np.reshape(ret, ret.shape + (1,))
    return ret
    
def load_img(path, format='.png'):
    scans = []
    for root, _, files in os.walk(path):
        for f in files:
            if f.find(format) != -1:
                scans.append(np.array(Image.open(os.path.join(root, f))))

    scans = np.array(scans)
    scans = np.reshape(scans, scans.shape + (1,))
    max_val = (2 ** 14) - 1
    scans = invert(scans, max_val)
    #scans = normalize(scans)
    scans = scans / max_val

    return scans

def trainGenerator(images, masks, batch_size, aug_dict, image_color_mode="grayscale",
                   mask_color_mode="grayscale", image_save_prefix="image", mask_save_prefix="mask",
                   flag_multi_class=False, num_class=2, save_to_dir=None, target_size=(256,256), seed=1, dcm=True):

    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)

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
        save_prefix  = mask_save_prefix,
        seed = seed)

    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        yield (img, mask)

def testGenerator(test, target_size=(256,256), flag_multi_class=False, as_gray=True, dcm=True):
    for s in test:
        img = np.reshape(s, (1,) + s.shape)
        yield img

if __name__ == '__main__':
    batch_size = 1
    epochs = 3
    data_aug = False
    steps_per_epoch = 20
    target_size = (256, 256)
    data_gen_args = dict(rotation_range=0.2,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         shear_range=0.05,
                         zoom_range=0.05,
                         horizontal_flip=False,
                         fill_mode='nearest')

    if len(os.listdir('data/resized_input')) <= 1:
        print('Loading scans')
        images = load_scans('data/input', target_size, save=('data/resized_input', 'png'))
    else:
        images = load_img('data/resized_input')

    if len(os.listdir('data/resized_masks')) <= 1:
        print('Loading masks')
        masks = load_scans('data/masks', target_size, is_mask=True, save=('data/resized_masks', 'png'))
    else:
        masks = load_img('data/resized_masks')

    X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.2)
    #X_test = [np.array(Image.open('data/0_0.1_425_1.0_0.1_1.0_1.0_2.0_deformed2/_0.dcm.png'))] # test with specific scan
    #X_test = [dicom.dcmread('data/2_0.2_1700_50.0_0.01_1.0_1.0_4.0_deformed/_0.dcm').pixel_array] # test with specific scan
    #max_val = (2 ** 14) - 1
    #X_test[0] = X_test[0] / max_val
    #X_test[0] = normalize(X_test[0])
    #X_test[0] = cv2.resize(X_test[0], target_size, interpolation=cv2.INTER_AREA)
    #X_test = np.array(X_test)
    #X_test = np.reshape(X_test, X_test.shape + (1,))
    #y_test = [np.array(Image.open('data/0_0.1_425_1.0_0.1_1.0_1.0_2.0_deformed_mask2/_0.dcm.png'))]
    #y_test = [dicom.dcmread('data/0_0.1_425_1.0_0.1_1.0_1.0_2.0_deformed_mask2/_0.dcm.png').pixel_array] # test with specific scan
    #y_test[0] = normalize(y_test[0])
    #y_test[0] = cv2.resize(y_test[0], target_size, interpolation=cv2.INTER_AREA)
    #y_test = np.array(y_test)
    #y_test = np.reshape(y_test, y_test.shape + (1,))

    gc.collect() # collect unused memory. hopefully.

    #myGene = trainGenerator(X_train, y_train, batch_size=batch_size, target_size=target_size, 
    #                        aug_dict=data_gen_args, save_to_dir=None)
    #testGene = testGenerator(X_test)
    model = unet()
    weights_path = f'unet_epochs_{epochs}_steps_{steps_per_epoch}.hdf5'
    loaded_weights = False
    if os.path.exists(weights_path):
        loaded_weights = True
        model.load_weights(weights_path)
    model_checkpoint = ModelCheckpoint(weights_path, monitor='accuracy',verbose=1, save_best_only=True)

    model.summary()
    #callbacks = [model_checkpoint]
    callbacks = []
    #if not loaded_weights:
    #    model.fit_generator(myGene, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=callbacks)
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks)

    lr_scheduler = LearningRateScheduler(lr_sch)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)

    num_tests = len(X_test)
    #predict = model.predict_generator(testGene, num_tests, verbose=1)
    predict = model.predict(X_test, batch_size=batch_size)

    for p in predict:
        show_img(p, method='cv2')
