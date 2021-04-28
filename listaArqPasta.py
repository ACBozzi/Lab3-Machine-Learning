import os, glob
import time

home = '/home/carol/Área de Trabalho/ML/LAB-3/data/train'

arquivo = open("train.txt", "a")
if os.path.isdir(home): #se existe
	os.chdir(home) #então abre
	for file in glob.glob('*.jpg'): #criar listas de arquivos a partir de buscas em diretórios
		classe =file.split('-')[0]
		nome = file
		print(nome+' '+classe)
		arquivo.write(nome+' '+classe+'\n')