import glob
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import csv
import sklearn as sk
from scipy import misc, ndimage

def get_names_from_folder(filepath, filetype='jpeg'):
    return glob.golb(filepath+filetype)

def generate_label(file_name):
    if 'ALB' in file_name:
        return [1,0,0,0,0,0,0,0]
    elif 'BET' in file_name:
        return [0,1,0,0,0,0,0,0]
    elif 'DOL' in file_name:
        return [0,0,1,0,0,0,0,0]
    elif 'LAG' in file_name:
        return [0,0,0,1,0,0,0,0]
    elif 'NoF' in file_name:
        return [0,0,0,0,1,0,0,0]
    elif 'OTHER' in file_name:
        return [0,0,0,0,0,1,0,0]
    elif 'SHARK' in file_name:
        return [0,0,0,0,0,0,1,0]
    else:
        return [0,0,0,0,0,0,0,1]


def generate_labels_from_images(images):
    labels = []
    for im in images:
        labels.append(generate_label(im))
    return labels

def get_filenames(directory):
    return glob.glob(directory+"*.jpg")

def get_directories():
    fALB = "../input/train/ALB/"
    fBET = "../input/train/BET/"
    fDOL = "../input/train/DOL/"
    fLAG = "../input/LAG/"
    fNoF = "../input/train/NoF/"
    fOTHER = "../input/train/OTHER/"
    fSHARK = "../input/train/SHARK/"
    fYFT = "../input/train/YFT/"
    return [fALB, fBET, fLAG, fNoF, fOTHER, fSHARK, fYFT]

def generate_list_of_images():
    images = []
    directories = get_directories()
    for d in directories:
        images = images + get_filenames(d)
    return images

def augment_image(image, augment):

    if augment == 'mirror':
        return np.fliplr(image)
    elif augment == 'rotate':
        rot = np.random.randint(10, 25)
        return ndimage.rotate(image, rot, reshape=False)
    elif augment == 'blurr:':
        return ndimage.guassian_filter(iamge, sigma=2)
    else:
        return image

def gen_images_labels_augs(image_list, label_list, augment=False, num_augs_per_img=None):

    images = []
    labels = []
    augments = []
    augmentations = ['none', 'mirror', 'blurr', 'rotate']

    if augment and num_augs_per_img == None:
        for i in range(len(image_list)):
            for j in range(len(augmentations)):
                images.append(image_list[i])
                labels.append(label_list[i])
                augments.append(augmentations[j])
    elif augment:
        for i in range(len(image_list)):
            for j in range(num_augs_per_img):
                if j == 0:
                    images.append(image_list[i])
                    labels.append(label_list[i])
                    augments.append(augmentations[0])
                else:
                    images.append(image_list[i])
                    labels.append(label_list[i])
                    augments.append(random.choice(augmentations[1:]))
    else:
        images = image_list
        labels = label_list
        augments = ['none']*len(image_list)

    return images, labels, augments

def image_to_mem(image, width, height, greyscale=False):
    tempimg = misc.imread(image, flatten=greyscale)
    tempimg = misc.imresize(tempimg, [width,height])
    return tempimg

def normalize_image(image):
    normalized_pixel_val = np.float32(1./255)
    image = np.multiply(image.flatten(), normalized_pixel_val)
    return image

def imagelist_to_mem(images, width, height, greyscale=False):
    images_rgb = []
    for im in images:
        images_rgb.append(image_to_mem(im, width, height,greyscale=greyscale))
    return images_rgb

def apply_image_augs(images, augs, greyscale):
    image_list = []
    for i in range(len(images)):
        image_list.append(augment_image(images[i], augs[i]))
    return image_list

def tf_image_prep(images, greyscale, imgsize=[32,32], augs=None):
    imgs = imagelist_to_mem(images, imgsize[0], imgsize[1], greyscale)
    if augs:
        imgs = apply_image_augs(imgs, augs, greyscale)
    images = []
    for im in imgs:
        images.append(normalize_image(im))
    return images

def shuffle_training_data(images, labels, augs=None, seed=0):
    if augs:
        images, labels, augs = sk.utils.shuffle(images, labels, augs, random_state=seed)
        return images, labels, augs
    else:
        images, labels = sk.utils.shuffle(images, labels, random_state=seed)
        return images, labels

def load_classification_images(testing=False, greyscale=False):
    directory = "../input/test_stg1/"
    image_filenames = get_filenames(directory)
    #images = imagelist_to_mem(images, 32, 32)
    if testing:
        image_filenames = image_filenames[:50]
    images = np.array(tf_image_prep(image_filenames, greyscale))
    return image_filenames, images

def prepare_data(testing=False, imgsize=[32,32], greyscale=False):
    images = generate_list_of_images()
    labels = generate_labels_from_images(images)
    images, labels, augs = gen_images_labels_augs(images, labels, augment=True)
    if testing:
        images, labels, augs = shuffle_training_data(images, labels, augs)
        images = images[:100]
        labels = labels[:100]
        augs = augs[:100]
    images = tf_image_prep(images, greyscale, augs=augs)
    return np.array(images), np.array(labels)

def create_tests(data, num_tests):
    index = len(images)-num_test
    data_tests = images[index:]
    data = images[:index-1]
    return data, data_tests

def prepare_data_with_tests(num_test, testing=False, imgsize=[32,32], greyscale=False):
    images, labels = prepare_data(testing=False, imgsize=[32,32], greyscale=False)
    images, image_tests = create_tests(images, num_tests)
    labels, label_tests = create_tests(labels, num_tests)
    return np.array(images), np.array(labels)

def gen_batch(images, labels, batch_size, index):

    if index+batch_size < len(images):
        batch_images = images[index:index+batch_size]
        batch_labels = labels[index:index+batch_size]
        return batch_images, batch_labels, index+batch_size
    else:
        batch_images = images[index:index+batch_size]
        batch_labels = labels[index:index+batch_size]

        for i in range(index+batch_size-len(images)):
            r = random.randint(0, len(images)-1)
            np.append(batch_images, images[r])
            np.append(batch_labels, labels[r])

        return batch_images, batch_labels, index+batch_size
