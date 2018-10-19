
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
from sklearn import cross_vali
from sklearn.svm import SVCdation


method = 'train'
path = 'train.txt'

if 'train' in method:
	train_data = load_data(path,method)
	train_data['shan'] = train_data['domain'].map(lambda  x:getshan(x)).astype(float)
	train_data['yuan_ratio'] = train_data['domain'].map(lambda x:getyuanyin(x)).astype(float)
	train_data['root'] = train_data['domain'].map(lambda x:getroot(x)).astype(str)
	train_data['rootclass'] = train_data['root'].map(lambda x:getrootclass(x)).astype(int)
	train_data['len'] = train_data['domain'].map(lambda x:getlen(x)).astype(int)

	scaler = preprocessing.StandardScaler()
	len_scale_param = scaler.fit(train_data['len'].values.reshape(-1,1))
	train_data['len_scaled'] = scaler.fit_transform(train_data['len'].values.reshape(-1,1),len_scale_param)
	shan_scale_param = scaler.fit(train_data['shan'].values.reshape(-1,1))
	train_data['shan_sclaed'] = scaler.fit_transform(train_data['shan'].values.reshape(-1,1),shan_scale_param)


	train_pre = train_data.filter(regex = 'label|yuan_ratio|len_scaled|shan_scale_param|rootclass')
	train_pre = shuffle(train_pre)
	print (train_data.info())
	print (train_data.head(10))
	print (train_data.tail(10))
	train_pre =train_pre.as_matrix()
	y_train = train_pre[0:200000,0]
	x_train = train_pre[0:200000,1:]
	y_test = train_pre[200000:201000,0]
	x_test = train_pre[200000:201000,1:]
	print ('now training')
	lr = SVC(kernel='rbf',C=0.4).fit(x_train,y_train)
	print ('training finished')
	model = cross_validation.cross_val_score(lr,x_train,y_train,cv = 5)
	print (model)







