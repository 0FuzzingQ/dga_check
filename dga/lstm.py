import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import argparse
from load import load_data
from common import getshan,getyuanyin,getroot,getrootclass,getlen,getw2v
from sklearn.feature_extraction.text import CountVectorizer
import re
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn import cross_validation
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense,Embedding
from keras.layers import LSTM


method = 'train'
path = 'train.txt'

if 'train' in method:
	train_data = load_data(path,method)
	train_data = shuffle(train_data)
	w2v_word_list,label_list = getw2v(train_data['domain'],train_data['label'])
	#print (type(w2v_word_list))
	#exit()
	x_train = w2v_word_list[0:80000]
	y_train = label_list[0:80000]
	x_test = w2v_word_list[80000:]
	y_test = label_list[80000:]
	model = Sequential()
	#model.add(Embedding())
	model.add(LSTM(128,dropout = 0.2,recurrent_dropout = 0.2))
	model.add(Dense(1,activation='sigmoid'))

	model.compile(loss = 'binary_crossentropy',optimizer = 'adam',metrics = ['accuracy'])
	print ('now training....')
	model.fit(x_train,y_train,nb_epoch = 50,batch_size = 32)
	print ('now evaling....')
	score,acc = model.evaluate(x_test,y_test)
	print (score,acc)

