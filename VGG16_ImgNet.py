
from keras.models import Sequential, load_model
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.preprocessing import image
from keras.layers.core import Flatten, Dense, Dropout
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
import numpy as np
np.set_printoptions(threshold=1000)
from keras.utils.data_utils import get_file
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from keras.utils.visualize_util import plot



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
del train_index


y_train = np.array([np.repeat(i,750) for i in range(10)])
y_train = y_train.reshape((7500,1))
y_train = to_categorical(y_train, 10)



'''

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
del test_index


y_test = np.array([np.repeat(i,250) for i in range(10)])
y_test = y_test.reshape((2500,1))
y_test = to_categorical(y_test, 10)

'''




# Model

model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(256, 256, 3)))

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

TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', TF_WEIGHTS_PATH_NO_TOP, cache_subdir='models')
model.load_weights(weights_path)

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


for layer in model.layers[:31]:
    layer.trainable = False





# compile
sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.8, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['categorical_accuracy'])


# Train
h = model.fit(x_train, y_train, batch_size=16, nb_epoch=50, verbose=1, shuffle= True, sample_weight=None)

model.save('VGG16_base')

np.savetxt('VGG16_loss', np.array(h.history['loss']))
np.savetxt('VGG16_acc', np.array(h.history['categorical_accuracy']))










'''

# load previous model
model = load_model('VGG16_base')



# Test
score = model.evaluate(x_test, y_test, batch_size=16, verbose=1)
print 'loss, accuracy: ', score

plot(model, to_file='VGG16_base.png')

'''
















