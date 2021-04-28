import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from PIL import Image
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.models import Model
from keras.layers import GlobalAveragePooling2D
from keras.preprocessing import image



#caso use o colab usar o google driva
drive_path = '/home/carol/Área de Trabalho/ML/LAB-3/'

## Classes - meses do ano
num_classes = 12

## Batch Size
batch_size = 64

## Epochs - número de épocas que vai roda
n_epochs = 16

## Train and Test files
train_file = drive_path + 'train.txt'
test_file = drive_path + 'test.txt'

## Input Image Dimension
img_rows, img_cols = 64, 64

print('Done')

## Resize

def resize_data(data, size, convert):

	if convert:
		data_upscaled = np.zeros((data.shape[0], size[0], size[1], 3))
	else:
		data_upscaled = np.zeros((data.shape[0], size[0], size[1]))
	for i, img in enumerate(data):
		large_img = cv2.resize(img, dsize=(size[1], size[0]), interpolation=cv2.INTER_CUBIC)
		data_upscaled[i] = large_img

	#print (np.shape(data_upscaled))
	return data_upscaled
  
print('Done')

## Load Images

def load_images(image_paths, convert=False):

	x = []
	y = []
	for image_path in image_paths:

		path, label = image_path.split(' ')
		
		## Image path
		path= drive_path + 'data/' + path
		print (path)

		if convert:
			image_pil = Image.open(path).convert('RGB') 
		else:
			image_pil = Image.open(path).convert('L')

		img = np.array(image_pil, dtype=np.uint8)

		x.append(img)
		y.append([int(label)])

	x = np.array(x)
	y = np.array(y)

	if np.min(y) != 0: 
		y = y-1

	return x, y

print('Done')

## Load Dataset

def load_dataset(train_file, test_file, resize, convert=False, size=(224,224)):

	arq = open(train_file, 'r')
	texto = arq.read()
	train_paths = texto.split('\n')
	
	print ('Size:', size)

	train_paths.remove('') # Remove empty lines
	train_paths.sort()

	print ("Loading training set...")
	x_train, y_train = load_images(train_paths, convert)
 
	arq = open(test_file, 'r')
	texto = arq.read()
	test_paths = texto.split('\n')

	test_paths.remove('') # Remove empty lines
	test_paths.sort()
 
	print ("Loading testing set...")
	x_test, y_test = load_images(test_paths, convert)

	if resize:
		print ("Resizing images...")
		x_train = resize_data(x_train, size, convert)
		x_test = resize_data(x_test, size, convert)

	if not convert:
		x_train = x_train.reshape(x_train.shape[0], size[0], size[1], 1)
		x_test = x_test.reshape(x_test.shape[0], size[0], size[1], 1)

	print (np.shape(x_train))
	return (x_train, y_train), (x_test, y_test)
 
print('Done')

print ("Loading database...")

## Gray Scale
#input_shape = (img_rows, img_cols, 1)
#(x_train, y_train), (x_test, y_test) = load_dataset(train_file, test_file, resize=True, convert=False, size=(img_rows, img_cols))

## RGB
input_shape = (img_rows, img_cols, 3)
(x_train, y_train), (x_test, y_test) = load_dataset(train_file, test_file, resize=True, convert=True, size=(img_rows, img_cols))

## Save for the confusion matrix
label = []
for i in range(len(x_test)):
	label.append(y_test[i][0])

## Normalize images
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
#print ('\n','x_train shape:', x_train.shape)

print ('\n',x_train.shape[0], 'train samples')
print ('\n',x_test.shape[0], 'test samples')


## Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

## Create CNN model
model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=1, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(2, 2), strides=1, activation='relu'))
model.add(MaxPooling2D(pool_size=(1,1)))
model.add(Conv2D(filters=32, kernel_size=(2, 2), strides=1, activation='relu'))
model.add(MaxPooling2D(pool_size=(1,1)))
model.add(Conv2D(filters=32, kernel_size=(2, 2), strides=1, activation='relu'))
model.add(MaxPooling2D(pool_size=(1,1)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
 
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


## Print CNN layers
print ('Network structure ----------------------------------')

# for i, layer in enumerate(model.layers):
# 	print(i,layer.name)
# 	if hasattr(layer, 'output_shape'):
# 		print(layer.output_shape)


print ('----------------------------------------------------')

## Configures the model for training
#model.compile(metrics=['accuracy'], loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(learning_rate=0.01))

## Trains the model
history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=n_epochs, verbose=2, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print ('\n----------------------------------------------------\n')
print ('Test loss:', score[0])
print ('Test accuracy:', score[1])
print ('\n----------------------------------------------------\n')

## Classes predicted
#print (model.predict_classes(x_test)) 

## Classes probability
#print (model.predict_proba(x_test)) 

pred = []
y_pred = model.predict_classes(x_test)
# y_pred = y_prob.argmax(axis=-1)
for i in range(len(x_test)):
	pred.append(y_pred[i])
print (confusion_matrix(label, pred))

acc = history.history['accuracy'] # history['acc'] / history['accuracy']
val_acc = history.history['val_accuracy'] # history['val_acc'] / history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
 
epochs = range(len(acc))


plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
 
plt.figure()
 
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
 
plt.show()


