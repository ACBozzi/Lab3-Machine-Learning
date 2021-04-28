from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
import numpy as np
import os
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import cv2


drive_path = '/home/carol/Área de Trabalho/ML/LAB-3/'
img_path = '/home/carol/Área de Trabalho/ML/LAB-3/data'
## Arquivo de entrada
entrada = drive_path + 'train.txt'

arq = open(entrada,'r')
conteudo_entrada = arq.readlines()
arq.close()

if os.path.isdir(img_path): #VERIFICA SE O CAMINHO PASSADO EXISTE
	os.chdir(img_path) #ABRE CASO EXISTA
	for conteudo in conteudo_entrada:
		# Initialising the ImageDataGenerator class. 
		# We will pass in the augmentation parameters in the constructor. 
		datagen = ImageDataGenerator( 
		        rotation_range = 8, 
		        shear_range = 0.2, 
		        zoom_range = 0.2,  
		        brightness_range = (0.5, 1.5)) 
		    
		img = load_img(conteudo.split(' ')[0])  
		nome = conteudo.split('.jpg')[0]
		clas = conteudo.split(' ')[1]
		clas = clas.strip('\n')
		x = img_to_array(img) 
		# Reshaping the input image 
		x = x.reshape((1, ) + x.shape)  
		   
		# Generating and saving 5 augmented samples  
		# using the above defined parameters.  
		i = 0
		for batch in datagen.flow(x, batch_size = 1, save_to_dir ='train', save_prefix =clas+'-', save_format ='jpg'): 
			i += 1
			if i > 2: 
				break
			#name = (nome'.jpg'+' '+clas)
			#arquivo.write(name)
