
# Merge-at-first Structure

from keras.models import Sequential, load_model
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.preprocessing import image
from keras.layers.core import Flatten, Dense, Dropout
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
import numpy as np
from keras.utils.data_utils import get_file





# Load labels
train_index = open('food-101/meta/train.txt').read().splitlines()
train_index = train_index[:7500]

'''
labels = open('food-101/meta/labels.txt').read().splitlines()
labels = labels[:10]

def to_label(y):
    l=[]
    for i in y:
        for j,k in enumerate(i):
            if k == 1: l.append(labels[j])
    return np.array(l)
'''

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


x_train = np.concatenate((x_train,x_train1),axis=3)
del x_train1



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





x_test1 = []
for i in test_index:
    img_path = 'food-101/sobel_images/' + i + '.jpg'
    img = image.load_img(img_path, target_size=(256, 256))
    x = image.img_to_array(img)
    x_test1.append(x)
x_test1 = np.array(x_test1)
x_test1 = pre_process(x_test1)
del test_index


x_test = np.concatenate((x_test,x_test1),axis=3)
del x_test1


np.save('xtrain6d', x_train)
np.save('xtest6d', x_test)







from keras.models import Sequential, load_model
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.preprocessing import image
from keras.layers.core import Flatten, Dense, Dropout
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
import numpy as np
from keras.utils.data_utils import get_file






y_train = np.array([np.repeat(i,750) for i in range(10)])
y_train = y_train.reshape((7500,1))
y_train = to_categorical(y_train, 10)


y_test = np.array([np.repeat(i,250) for i in range(10)])
y_test = y_test.reshape((2500,1))
y_test = to_categorical(y_test, 10)


x_train = np.load('xtrain6d.npy')
x_test = np.load('xtest6d.npy')




model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(256, 256, 6)))

model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

#TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
#weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', TF_WEIGHTS_PATH_NO_TOP, cache_subdir='models')
weights_path = 'VGG16_base'

model1 = Sequential()
model1.add(ZeroPadding2D((1,1),input_shape=(256, 256, 3)))

model1.add(Convolution2D(64, 3, 3, activation='relu'))
model1.add(ZeroPadding2D((1,1)))
model1.add(Convolution2D(64, 3, 3, activation='relu'))
model1.add(MaxPooling2D((2,2), strides=(2,2)))

model1.add(ZeroPadding2D((1,1)))
model1.add(Convolution2D(128, 3, 3, activation='relu'))
model1.add(ZeroPadding2D((1,1)))
model1.add(Convolution2D(128, 3, 3, activation='relu'))
model1.add(MaxPooling2D((2,2), strides=(2,2)))

model1.add(ZeroPadding2D((1,1)))
model1.add(Convolution2D(256, 3, 3, activation='relu'))
model1.add(ZeroPadding2D((1,1)))
model1.add(Convolution2D(256, 3, 3, activation='relu'))
model1.add(ZeroPadding2D((1,1)))
model1.add(Convolution2D(256, 3, 3, activation='relu'))
model1.add(MaxPooling2D((2,2), strides=(2,2)))

model1.add(ZeroPadding2D((1,1)))
model1.add(Convolution2D(512, 3, 3, activation='relu'))
model1.add(ZeroPadding2D((1,1)))
model1.add(Convolution2D(512, 3, 3, activation='relu'))
model1.add(ZeroPadding2D((1,1)))
model1.add(Convolution2D(512, 3, 3, activation='relu'))
model1.add(MaxPooling2D((2,2), strides=(2,2)))

model1.add(ZeroPadding2D((1,1)))
model1.add(Convolution2D(512, 3, 3, activation='relu'))
model1.add(ZeroPadding2D((1,1)))
model1.add(Convolution2D(512, 3, 3, activation='relu'))
model1.add(ZeroPadding2D((1,1)))
model1.add(Convolution2D(512, 3, 3, activation='relu'))
model1.add(MaxPooling2D((2,2), strides=(2,2)))

model1.add(Flatten())
model1.add(Dense(4096, activation='relu'))
model1.add(Dropout(0.5))
model1.add(Dense(4096, activation='relu'))
model1.add(Dropout(0.5))
model1.add(Dense(10, activation='softmax'))

model1.load_weights(weights_path)





for i in range(2,37):
    model.layers[i].set_weights(model1.layers[i].get_weights())
    model.layers[i].trainable = False


del model1


for layer in model.layers:
    layer.trainable = True



model.load_weights('new')

# compile
sgd = SGD(lr=1e-7, decay=1e-4, momentum=0.8, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['categorical_accuracy'])


#calls = [ModelCheckpoint('VGG16_edge',monitor='val_acc', verbose=1, save_best_only=True), EarlyStopping(monitor='val_acc', min_delta=0, patience=1, verbose=1, mode='auto')]


# Train
h = model.fit(x_train, y_train, batch_size=8, nb_epoch=1, verbose=1, shuffle= True, validation_data=(x_test,y_test),sample_weight=None)





np.savetxt('new_loss', np.array(h.history['loss']))
np.savetxt('new_acc', np.array(h.history['categorical_accuracy']))
np.savetxt('new_vloss', np.array(h.history['val_loss']))
np.savetxt('new_vacc', np.array(h.history['val_categorical_accuracy']))
# save
model.save('new')















