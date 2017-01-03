
# For edge image transformation


import numpy as np
from skimage.io import imread, imshow, imshow_collection, imsave
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage.transform import resize
import os


labels = open('food-101/meta/classes.txt').read().splitlines()
labels = labels[:10]
for i in labels:
    if not os.path.exists('food-101/sobel_images/' + i):
        os.makedirs('food-101/sobel_images/' + i)


train_index = open('food-101/meta/train.txt').read().splitlines()
train_index = train_index[:7500]

n=0
for i in train_index:
    img_path = 'food-101/images/' + i + '.jpg'
    img = imread(img_path, as_grey=False)
    if len(img.shape) == 2:
        img = sobel(img)
    else:
        img[:,:,0] = sobel(img[:,:,0]) * 255
        img[:,:,1] = sobel(img[:,:,1]) * 255
        img[:,:,2] = sobel(img[:,:,2]) * 255
    imsave('food-101/sobel_images/' + i + '.jpg', img)


test_index = open('food-101/meta/test.txt').read().splitlines()
test_index = test_index[:2500]

n=0
for i in test_index:
    img_path = 'food-101/images/' + i + '.jpg'
    img = imread(img_path, as_grey=False)
    if len(img.shape) == 2:
        img = sobel(img)
    else:
        img[:,:,0] = sobel(img[:,:,0]) * 255
        img[:,:,1] = sobel(img[:,:,1]) * 255
        img[:,:,2] = sobel(img[:,:,2]) * 255
    imsave('food-101/sobel_images/' + i + '.jpg', img)


'''

# Example figures

l = []

d = imread('food-101/images/fried_calamari/165952.jpg', as_grey=False)
d[:,:,0] = roberts(d[:,:,0]) * 255
d[:,:,1] = roberts(d[:,:,1]) * 255
d[:,:,2] = roberts(d[:,:,2]) * 255
l.append(d)
d = imread('food-101/images/fried_calamari/165952.jpg', as_grey=False)
d[:,:,0] = sobel(d[:,:,0]) * 255
d[:,:,1] = sobel(d[:,:,1]) * 255
d[:,:,2] = sobel(d[:,:,2]) * 255
l.append(d)
d = imread('food-101/images/fried_calamari/165952.jpg', as_grey=False)
d[:,:,0] = scharr(d[:,:,0]) * 255
d[:,:,1] = scharr(d[:,:,1]) * 255
d[:,:,2] = scharr(d[:,:,2]) * 255
l.append(d)
d = imread('food-101/images/fried_calamari/165952.jpg', as_grey=False)
d[:,:,0] = prewitt(d[:,:,0]) * 255
d[:,:,1] = prewitt(d[:,:,1]) * 255
d[:,:,2] = prewitt(d[:,:,2]) * 255
l.append(d)
imshow_collection(l)




'''















