import pydicom as dicom
import numpy as np
import cv2
from nn import *
import os
from PIL import Image, ImageFilter
from sklearn.model_selection import train_test_split
import gc
import time

max_14b = float((2 ** 14) - 1)
max_16b = float((2 ** 16) - 1)

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

def normalize(img, int8=False, max_val=None):
    img = img / np.max(img) if max_val is None else img / max_val
    
    if int8:
        img = (img * 255).astype(np.uint8)

    return img

def binarize(img, thresh=0.5, max_val=1.0):
    img[img > thresh] = max_val
    img[img <= thresh] = 0.0

def invert(img, max_val=1.0):
    ones = np.ones_like(img) * max_val

    return ones - img

def show_img(img, method='cv2', norm=False):
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

        aux = aux / max_14b
        #aux = crop_bg(aux)

        if is_mask:
            aux = invert(aux)
            thresh = 0.61
            binarize(aux, thresh)

        aux = cv2.resize(aux, target_size, interpolation=cv2.INTER_AREA)
        ret.append(aux)

    if save is not None:
        print('Saving scans')

        for (s, r) in zip(scans, ret):
            split = s.split('/')
            path = ''
            for p in split:
                if p.find('.dcm') == -1 and p not in ['data', 'input', 'masks']:
                    path += p
            name = split[-1]
            os.makedirs(f'{save[0]}/{path}', exist_ok=True)
            if not is_mask:
                cv2.imwrite(f'{save[0]}/{path}/{name}.{save[1]}', (r * max_14b).astype(np.uint16))
            else:
                cv2.imwrite(f'{save[0]}/{path}/{name}.{save[1]}', (r * max_16b).astype(np.uint16))

    ret = np.array(ret)
    ret = np.reshape(ret, ret.shape + (1,))
    return ret
    
def load_img(path, format='.png', is_mask=False):
    scans = []
    for root, _, files in os.walk(path):
        for f in files:
            if f.find(format) != -1:
                scans.append(np.array(cv2.imread(os.path.join(root, f), -1)))

    scans = np.array(scans)
    scans = np.reshape(scans, scans.shape + (1,))

    return scans

def train_generator(images, masks, batch_size, aug_dict, image_color_mode="grayscale",
                   mask_color_mode="grayscale", image_save_prefix="image", mask_save_prefix="mask",
                   flag_multi_class=False, num_class=2, save_to_dir=None, target_size=(256,256), seed=1, dcm=True):

    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)

    image_generator = image_datagen.flow(
        images,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed)

    mask_generator = mask_datagen.flow(
        masks,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed)

    for (img, mask) in zip(image_generator, mask_generator):
        yield (img, mask)

def test_generator(test, target_size=(256,256), flag_multi_class=False, as_gray=True, dcm=True):
    for s in test:
        img = np.reshape(s, (1,) + s.shape)
        yield img

if __name__ == '__main__':
    testing = False
    batch_size = 4
    epochs = 20
    data_aug = True
    steps_per_epoch = 20
    target_size = (256, 256)
    data_gen_args = dict(rotation_range=0.2,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         shear_range=0.05,
                         zoom_range=0.05,
                         horizontal_flip=True,
                         fill_mode='nearest')

    weights_path = f'unet_b_{batch_size}_e_{epochs}_s_{steps_per_epoch}_t_{target_size[0]}x{target_size[1]}.hdf5'
    loaded_weights = False
    if not os.path.exists(weights_path) or not testing:
        if len(os.listdir('data/resized_input')) <= 1:
            print('Loading scans')
            start = time.time()
            X_train = load_scans('data/input', target_size, save=('data/resized_input', 'png'))
            print(f'Took {time.time() - start}s to complete')
        else:
            print('Loading input images')
            X_train = load_img('data/resized_input')
            X_train = X_train / max_14b

        if len(os.listdir('data/resized_masks')) <= 1:
            print('Loading masks')
            start = time.time()
            y_train = load_scans('data/masks', target_size, is_mask=True, save=('data/resized_masks', 'png'))
            print(f'Took {time.time() - start}s to complete')
        else:
            print('Loading mask images')
            y_train = load_img('data/resized_masks')
            y_train = y_train / max_16b

    else:
        print('Loading saved weights')
        loaded_weights = True

    #X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)
    X_test = [dicom.dcmread('data/test/test3_25/0_0.1_425_25.0_0.1_1.0_1.0_2.0_deformed.dcm').pixel_array,
              dicom.dcmread('data/test/test3_25/0_0.2_425_25.0_0.1_1.0_1.0_2.0_deformed.dcm').pixel_array,
              dicom.dcmread('data/test/test4_50/1_0.1_850_50.0_0.01_1.0_1.0_4.0_deformed.dcm').pixel_array,
              dicom.dcmread('data/test/test4_50/1_0.2_850_50.0_0.01_1.0_1.0_4.0_deformed.dcm').pixel_array
             ]

    y_test = [dicom.dcmread('data/test/test3_25/0_0.1_425_25.0_0.1_1.0_1.0_2.0_deformed_mask.dcm').pixel_array,
              dicom.dcmread('data/test/test3_25/0_0.2_425_25.0_0.1_1.0_1.0_2.0_deformed_mask.dcm').pixel_array,
              dicom.dcmread('data/test/test4_50/1_0.1_850_50.0_0.01_1.0_1.0_4.0_deformed_mask.dcm').pixel_array,
              dicom.dcmread('data/test/test4_50/1_0.2_850_50.0_0.01_1.0_1.0_4.0_deformed_mask.dcm').pixel_array
             ]

    for i in range(len(X_test)):
        X_test[i] = X_test[i] / max_14b
        X_test[i] = cv2.resize(X_test[i], target_size, interpolation=cv2.INTER_AREA)

    for i in range(len(y_test)):
        y_test[i] = y_test[i] / max_14b
        y_test[i] = invert(y_test[i])
        thresh = 0.61
        binarize(y_test[i], thresh)
        y_test[i] = cv2.resize(y_test[i], target_size, interpolation=cv2.INTER_AREA)

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, X_test.shape + (1,))
    y_test = np.array(y_test)
    y_test = np.reshape(y_test, y_test.shape + (1,))

    #gc.collect() # collect unused memory. hopefully.

    model = unet(input_size=target_size+(1,))

    model.summary()

    if loaded_weights and testing:
        model.load_weights(weights_path)
    else:
        model_checkpoint = ModelCheckpoint(weights_path, monitor='loss', verbose=1, save_best_only=True)
        #callbacks = []
        callbacks = [model_checkpoint]
        train_gen = train_generator(X_train, y_train, batch_size=batch_size, target_size=target_size,
                                    aug_dict=data_gen_args, save_to_dir=None)
        model.fit_generator(train_gen, steps_per_epoch=len(X_train) / batch_size, epochs=epochs, callbacks=callbacks)
        #model.fit(X_train, y_train, callbacks=callbacks, epochs=epochs, batch_size=batch_size)

    lr_scheduler = LearningRateScheduler(lr_sch)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)

    num_tests = len(X_test)

    for (x, y) in zip(X_test, y_test):
        x = np.reshape(x, (1,) + x.shape)
        y = np.reshape(y, (1,) + y.shape)
        test_loss, test_acc = model.evaluate(x, y, batch_size=batch_size)
        print(f'Test loss = {test_loss}, test acc = {test_acc}')

    predict = model.predict(X_test)

    i = 0
    for p in predict:
        save_img((p * max_16b).astype(np.uint16), f'{i}.png', method='cv2')
        #show_img(y_test[i], method='cv2')
        i += 1
