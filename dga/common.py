import numpy as np
import pandas as pd
import math
from hmmlearn import hmm 
from gensim.models import Word2Vec

def getshan(domain):
	tmp_dict = {}
	domain_len = len(domain)
	for i in range(0,domain_len):
		if domain[i] in tmp_dict.keys():
			tmp_dict[domain[i]] = tmp_dict[domain[i]] + 1
		else:
			tmp_dict[domain[i]] = 1
	shannon = 0
	for i in tmp_dict.keys():
		p = float(tmp_dict[i]) / domain_len
		shannon = shannon - p * math.log(p,2)
	return shannon

def getyuanyin(domain):
	yuan_list = ['a','e','i','o','u']
	domain = domain.lower()
	count_word = 0
	count_yuan = 0
	yuan_ratio = 0
	for i in range(0,len(domain)):
		if ord(domain[i]) >= ord('a') and ord(domain[i]) <= ord('z'):
			count_word = count_word + 1
			if domain[i] in yuan_list:
				count_yuan = count_yuan + 1
	if count_word == 0:
		return yuan_ratio
	else:
		yuan_ratio = float(count_yuan) / count_word
		return yuan_ratio

def getroot(domain):
	return domain.split('.')[-1]

def getrootclass(root):
	common_root = ['cn','com','cc','net','org','gov','info']
	if root in common_root:
		return 0
	else:
		return 1

def getlen(domain):
	return len(domain)

def getw2v(domain_list,label_list):
	stop = ['/','~']
	w2v_list = []
	for i in range(0,domain_list.size):
		tmp = []
		name = domain_list[i].split('.')[0]
		for j in range(0,len(name)):
			tmp.append(name[j])
		w2v_list.append(tmp)

	model = Word2Vec(w2v_list, min_count = 1)
	#print (model._vocabulary)
	model.wv.save_word2vec_format('word2vec.txt',binary=False) 
	#print model['a']
	label_vect = []
	wv_vect = []
	for i in range(0,domain_list.size):
		name = domain_list[i].replace('.','')
		tmp = []
		vect = []
		for j in range(0,len(name)):
			if name[j] in stop:
				continue
			tmp.append(model[name[j]])
			if j >= 9:
				break
		if len(tmp) < 10:
			for k in range(0,10-len(tmp)):
				tmp.append([0]*100)
		vect = np.vstack((x for x in tmp))
		wv_vect.append(vect)
		label_vect.append(label_list[i])
		if i ==100000:
			break
	wv_vect = np.array(wv_vect)
	label_vect = np.array(label_vect)
	return wv_vect,label_vect
		

		

