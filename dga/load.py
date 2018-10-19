# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

def load_data(path,method):
	if method == 'train':
		url_list = []
		label_list = []
		
		with open(path,'r') as f:
			content = f.readlines()
			for i in range(0,len(content)):
				tmp = content[i].strip('\n').lower()
				#print tmp.split(' ')
				url_list.append(tmp.split('\t')[0].strip()[0:-1])
				label_list.append(tmp.split('\t')[1].strip())
		f.close()
		
		#print 'can not read file'
		
		tmp = np.vstack((label_list,url_list))
		train_data = pd.DataFrame(tmp.T,columns = ['label','domain'])
		#print train_data.info()
		#print train_data.head()
		return train_data

	elif method == 'test':
		exit()

#load_data('train.txt','train')
