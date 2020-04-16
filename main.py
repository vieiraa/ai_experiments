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

        #CLAHE APICATION
        #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

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
            cv2.imwrite(f'{save[0]}/{path}/{name}.{save[1]}', (aux * 255).astype(np.uint8))

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
    scans = normalize(scans)

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
        save_prefix  = image_save_prefix,
        seed = seed)

    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        yield (img, mask)

def testGenerator(test, target_size=(256,256), flag_multi_class=False, as_gray=True, dcm=True):
    for s in test:
        img = np.reshape(s, (1,) + s.shape)
        yield img

if __name__ == '__main__':
    batch_size = 4
    epochs = 150
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

    if len(os.listdir('data/resized_input')) <= 1:
        print('Loading scans')
        images = load_scans('data/input', target_size)#, save=('data/resized_input', 'png'))
    else:
        print('Loading input images')
        images = load_img('data/resized_input')

    if len(os.listdir('data/resized_masks')) <= 1:
        print('Loading masks')
        masks = load_scans('data/masks', target_size, is_mask=True)#, save=('data/resized_masks', 'png'))
    else:
        print('Loading mask images')
        masks = load_img('data/resized_masks')

    X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.2)
    X_test = [dicom.dcmread('data/test/test3_25/0_0.1_425_25.0_0.1_1.0_1.0_2.0_deformed.dcm').pixel_array,
              dicom.dcmread('data/test/test3_25/0_0.2_425_25.0_0.1_1.0_1.0_2.0_deformed.dcm').pixel_array,
              dicom.dcmread('data/test/test4_50/1_0.1_850_50.0_0.01_1.0_1.0_4.0_deformed.dcm').pixel_array,
              dicom.dcmread('data/test/test4_50/1_0.2_850_50.0_0.01_1.0_1.0_4.0_deformed.dcm').pixel_array]

    y_test = [dicom.dcmread('data/test/test3_25/0_0.1_425_25.0_0.1_1.0_1.0_2.0_deformed_mask.dcm').pixel_array,
              dicom.dcmread('data/test/test3_25/0_0.2_425_25.0_0.1_1.0_1.0_2.0_deformed_mask.dcm').pixel_array,
              dicom.dcmread('data/test/test4_50/1_0.1_850_50.0_0.01_1.0_1.0_4.0_deformed_mask.dcm').pixel_array,
              dicom.dcmread('data/test/test4_50/1_0.2_850_50.0_0.01_1.0_1.0_4.0_deformed_mask.dcm').pixel_array]

    max_val = (2 ** 14) - 1
    for i in range(len(X_test)):
        X_test[i] = X_test[i] / max_val
        X_test[i] = cv2.resize(X_test[i], target_size, interpolation=cv2.INTER_AREA)

    for i in range(len(y_test)):
        y_test[i] = y_test[i] / max_val
        y_test[i] = cv2.resize(y_test[i], target_size, interpolation=cv2.INTER_AREA)

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, X_test.shape + (1,))
    y_test = np.array(y_test)
    y_test = np.reshape(y_test, y_test.shape + (1,))

    gc.collect() # collect unused memory. hopefully.

    myGene = trainGenerator(X_train, y_train, batch_size=batch_size, target_size=target_size, 
                            aug_dict=data_gen_args, save_to_dir=None)
    testGene = testGenerator(X_test)
    model = unet(input_size=target_size+(1,))
    weights_path = f'unet_epochs_{epochs}_steps_{steps_per_epoch}.hdf5'
    loaded_weights = False
    if os.path.exists(weights_path):
        loaded_weights = True
        model.load_weights(weights_path)
    model_checkpoint = ModelCheckpoint(weights_path, monitor='accuracy',verbose=1, save_best_only=True)

    model.summary()
    #callbacks = []
    callbacks = [model_checkpoint]
    if not loaded_weights:
        model.fit_generator(myGene, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=callbacks)

    lr_scheduler = LearningRateScheduler(lr_sch)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)

    num_tests = len(X_test)

    test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=batch_size)
    print(f'Test loss = {test_loss}, test acc = {test_acc}')

    predict = model.predict_generator(testGene, num_tests, verbose=1)

    i = 0
    for p in predict:
        show_img(p, method='cv2')
        show_img(y_test[i], method='cv2')
        i += 1
