import tensorflow as tf
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from numpy import *


path_test ='D:\python\homework\Attachment\Attachment 2'
path_pre='D:\python\homework\Attachment\Attachment 3'
CATEGORIES = []
for filename in os.listdir(path_test):
    CATEGORIES.append(filename)
IMG_SIZE = 100
training = []
for category in CATEGORIES:
    z = 1
    path = os.path.join(path_test, category)
    class_num = CATEGORIES.index(category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img))
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        new_array = cv2.cvtColor(new_array, cv2.COLOR_BGR2GRAY)
        training.append([new_array, class_num])
        z=z+1
        if z>700:
            break
random.shuffle(training)
pre=[]
for img_name in os.listdir(path_pre):
    img_path = path_pre + '\\' + img_name
    img = cv2.imread(img_path)
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img1 = cv2.resize(img1, (IMG_SIZE, IMG_SIZE))
    pre.append(img1)
X = []
y = []
for features, label in training:
    X.append(features)
    y.append(label)
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE)
pre = np.array(pre).reshape(-1, IMG_SIZE, IMG_SIZE)
X = X.astype('float32')
X /= 255
X = np.expand_dims(X,axis=3)
pre = pre.astype('float32')
pre /= 255
pre = np.expand_dims(pre,axis=3)

#y = np_utils.to_categorical(y, 5)
y=np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

batch_size = 16
#nb_classes = 4
nb_epochs = 3
#img_rows, img_columns = 200, 200
#img_channel = 3
#nb_filters = 32
#nb_pool = 2
#nb_conv = 3

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation=tf.nn.relu,
                           input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(5, activation=tf.nn.softmax)
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epochs, verbose=1)

score = model.evaluate(X_test, y_test, verbose=0)
print("Test Score: ", score[0])
print("Test accuracy: ", score[1])
pre_all=model.predict(pre)
result=np.argmax(pre_all,axis=1)+1

#color_mapping = {1: 'red', 2: 'green', 3: 'blue', 4: 'orange', 5: 'purple'}

# 提取每个元素对应的颜色
#colors = [color_mapping[val] for val in result]
#plt.bar(range(len(result)),result,color=colors)
#plt.xlabel('id')
#plt.ylabel('class')
#plt.title('Predicted Classes of Attachment 3')
#plt.show()

num=[]
for i in range(5):
    num.append(len(result[result==(i+1)]))
plt.bar(['Apple','Carambola','Pear','Plum','Tomato'],num,width=0.5)
plt.xlabel('Class')
plt.ylabel('num')
plt.title('Histogram of Classification')
plt.show()