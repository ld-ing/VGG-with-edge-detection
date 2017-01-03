
# Merge-at-last Structure

from keras.models import Sequential, load_model
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.preprocessing import image
from keras.layers.core import Flatten, Dense, Dropout
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from keras.engine.topology import Merge
import numpy as np
np.set_printoptions(threshold=1000)
from keras.utils.data_utils import get_file
from skimage.io import imread, imshow, imshow_collection, imsave
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage.transform import resize
import os
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping





# Load labels
train_index = open('food-101/meta/train.txt').read().splitlines()
train_index = train_index[:7500]
labels = open('food-101/meta/labels.txt').read().splitlines()
labels = labels[:10]

def to_label(y):
    l=[]
    for i in y:
        for j,k in enumerate(i):
            if k == 1: l.append(labels[j])
    return np.array(l)

def pre_process(x):
    x = x[:, :, :, ::-1]
    x[:, :, :, 0] -= 89.4
    x[:, :, :, 1] -= 112
    x[:, :, :, 2] -= 136.8
    return x

# Input images
x_train = []
for i in train_index:
    img_path = 'food-101/images/' + i + '.jpg'
    img = image.load_img(img_path, target_size=(256, 256))
    x = image.img_to_array(img)
    x_train.append(x)
x_train = np.array(x_train)
x_train = pre_process(x_train)



y_train = np.array([np.repeat(i,750) for i in range(10)])
y_train = y_train.reshape((7500,1))
y_train = to_categorical(y_train, 10)







def pre_process1(x):
    x = x[:, :, :, ::-1]
    x[:, :, :, 0] -= 8.65
    x[:, :, :, 1] -= 8.51
    x[:, :, :, 2] -= 8.29
    return x

# Input images
x_train1 = []
for i in train_index:
    img_path = 'food-101/sobel_images/' + i + '.jpg'
    img = image.load_img(img_path, target_size=(256, 256))
    x = image.img_to_array(img)
    x_train1.append(x)
x_train1 = np.array(x_train1)
x_train1 = pre_process1(x_train1)
del train_index







test_index = open('food-101/meta/test.txt').read().splitlines()
test_index = test_index[:2500]

x_test = []
for i in test_index:
    img_path = 'food-101/images/' + i + '.jpg'
    img = image.load_img(img_path, target_size=(256, 256))
    x = image.img_to_array(img)
    x_test.append(x)
x_test = np.array(x_test)
x_test = pre_process(x_test)


y_test = np.array([np.repeat(i,250) for i in range(10)])
y_test = y_test.reshape((2500,1))
y_test = to_categorical(y_test, 10)




x_test1 = []
for i in test_index:
    img_path = 'food-101/sobel_images/' + i + '.jpg'
    img = image.load_img(img_path, target_size=(256, 256))
    x = image.img_to_array(img)
    x_test1.append(x)
x_test1 = np.array(x_test1)
x_test1 = pre_process(x_test1)
del test_index













# load previous model
model1 = load_model('VGG16_base')
model1.pop()
model2 = load_model('VGG_edge')
model2.pop()

for layer in model1.layers:
    layer.trainable = True
for layer in model2.layers:
    layer.trainable = True

model_merged = Sequential()
model_merged.add(Merge([model1,model2], mode = 'concat', concat_axis=1))
model_merged.add(Dense(10, activation='softmax'))


# compile
sgd = SGD(lr=0.00001, decay=1e-6, momentum=0.8, nesterov=True)
model_merged.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['categorical_accuracy'])


#calls = [ModelCheckpoint('VGG16_edge',monitor='val_acc', verbose=1, save_best_only=True), EarlyStopping(monitor='val_acc', min_delta=0, patience=1, verbose=1, mode='auto')]


# Train
h = model_merged.fit([x_train,x_train1], y_train, batch_size=8, nb_epoch=20, verbose=1, shuffle= True, validation_data=([x_test,x_test1],y_test),sample_weight=None)
'''
np.savetxt('merge_loss', np.array(h.history['loss']))
np.savetxt('merge_acc', np.array(h.history['categorical_accuracy']))

# save
model_merged.save('VGG16_merged_f')




'''










