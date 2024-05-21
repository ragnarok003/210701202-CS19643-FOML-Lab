# %%
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.layers import LeakyReLU
from keras.models import Sequential
from keras.layers import Dense ,Conv2D, MaxPool2D ,Flatten, Dropout
from keras.datasets import mnist

# %%
mnist.load_data

# %%
(x_train, y_train), (x_test, y_test)=mnist.load_data()

x_train.shape ,y_train.shape ,x_test.shape ,y_test.shape

# %%
plt.imshow(x_train[0])

# %%
def plot_input_img(i):
   plt.imshow(x_train[i], cmap='binary')
   plt.title(y_train[i])
   # plt.axes('off')
   plt.show()

# %%
for i in range(10):
   plot_input_img(i)

# %%
#pre process the images

#normalizing the image to [0,1] range
x_train =x_train.astype(np.float32)/255
x_test =x_test.astype(np.float32)/255


#reshape /expand the dimention of images to (28,28,1)
x_train=np.expand_dims(x_train, -1)
x_test=np.expand_dims(x_test, -1)

#convert classes to one hot vectors
y_train=keras.utils.to_categorical(y_train)
y_test=keras.utils.to_categorical(y_test)

# %%
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(28,28,1)))
model.add(LeakyReLU(alpha=0.01))
model.add(MaxPool2D((2,2)))

model.add(Conv2D(64, (3,3)))
model.add(LeakyReLU(alpha=0.01))
model.add(MaxPool2D((2,2)))

model.add(Flatten())

model.add(Dropout(0.25))

model.add(Dense(10, activation="softmax"))


# %%
model.summary()

# %%
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# %%
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print(tf.config.experimental.list_physical_devices('GPU'))


# %%
import tensorflow as tf
tf.config.list_physical_devices("GPU")

# %%
model.compile(optimizer= 'adam',loss=keras.losses.categorical_crossentropy ,metrics=['accuracy'])

# %%
#Callbacks

from keras.callbacks import EarlyStopping, ModelCheckpoint

#Earlystopping

es=EarlyStopping(monitor='val_acc' ,min_delta=0.01, patience =4, verbose =1)

#Model check Point

mc=ModelCheckpoint("./foml.h5",moniter="val_acc" ,verbose=1,save_best_only= True )

cb = [es,mc]

# %%
#model training

his =model.fit(x_train, y_train, epochs= 5, validation_split=0.3,callbacks= cb)

# %%
model_S=keras.models.load_model("./foml.h5")

# %%
score =model_S.evaluate(x_test,y_test)
print(f' the model accuracy is {score[1]}')


