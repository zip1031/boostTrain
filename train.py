#
#
# Author : ZiP_fan <caofan1031@gmail.com>
#
# Time : 五 26  1 2018
#


import sys
import json
import xgboost as xgbt
import lightgbm as lgb
import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_auc_score 
encoding = 'utf-8'

def read_data(cfg):
	rf_num = cfg['rf_num']

	test_data = None
	test_label = None

	train_data = pd.read_csv(cfg['train_data'])
	train_label = pd.read_csv(cfg['train_label'])

	if cfg['rf_num'] <= 1:
		test_label = pd.read_csv(cfg['test_label'])
		test_data = pd.read_csv(cfg['test_data'])
	data = {}
	data['train_data'] = train_data
	data['test_data'] = test_data
	data['train_label'] = train_label
	data['test_label'] = test_label
	return data

def get_feature(cfg):
	feature_list_path = cfg['feature_path']
	features = pd.read_csv(feature_list_path)
	return list(features['feature_names'].values)

def prepare_data(data, rf_No, cfg):
	features = get_feature(cfg)
	
	train_data = None
	test_data = None
	train_label = None
	test_label = None

	if cfg['rf_num'] <= 1:
		train_data = np.array(data['train_data'][features].values, dtype = np.float32)
		test_data = np.array(data['test_data'][features].values, dtype = np.float32)
		train_label = np.array(data['train_label']['label'].values, dtype = np.float32)
		test_label = np.array(data['test_label']['label'].values, dtype = np.float32)
	else :
		all_data = data['train_data']
		all_label = data['train_label']
		group_num = cfg['rf_num']	
		group_by = cfg['rf_by']
		train_data = np.array(all_data[all_data[group_by] % group_num != rf_No][features].values, dtype = np.float32)
		test_data = np.array(all_data[all_data[group_by] % group_num == rf_No][features].values, dtype = np.float32)
		train_label = np.array(all_label[all_label[group_by] % group_num != rf_No]['label'].values, dtype = np.float32)
		test_label = np.array(all_label[all_label[group_by] % group_num == rf_No]['label'].values, dtype = np.float32)
	
	if cfg['lib'] == 'lgb':
		d_train = lgb.Dataset(train_data, train_label)
		d_test = lgb.Dataset(test_data, test_label)
	elif cfg['lib'] == 'xgb':
		d_train = xgbt.DMatrix(train_data, train_label)
		d_test = xgbt.DMatrix(test_data, test_label)
	
	test = {}
	test['data'] = test_data
	test['label'] = test_label
	print('Load Data Done')
	return d_train, d_test, test


def train(d_train, d_test, rf_No, cfg, test):
	params = cfg['params']
	num_round = cfg['num_round']
	early_stopping = cfg['early_stopping']
	model_path = os.path.join(cfg['model_saved_as'])
	
	if cfg['lib'] == 'lgb':
		print(str(rf_No) + " Train...")
		gbm = lgb.train(params, d_train, num_boost_round = num_round, valid_sets = [d_train, dtest])
		print(str(rf_No) + " Train Done!")
		gbm.save_model(model_path + '_' + str(rf_No) + '.model')
		
		features = get_feature(cfg)
		sorted_feature_importance_path = os.path.join(cfg['sorted_feature_importance']) if 'sorted_feature_importance' in cfg else None
		if sorted_feature_importance_path:
			fscore = list(gbm.feature_importance('gain'))
			feature_importance = zip(features, fscore)
			sorted_feature_importance = sorted(feature_importance, key = operator.itemgetter(1), reverse=True)
			with open(sorted_feature_importance_path + '_' + str(rf_No) + '.csv', 'w', encoding=encoding) as f:
				f.write('feature_name,f_score\n')
				for feature_name, f_score in sorted_feature_importance:
					f.write('{0},{1}\n'.format(feature_namgit@gitlab.yangqianguan.com:backend/yqg_core.gite, f_score))
		score = lgb.predict(test['data'])
		auc = roc_auc_score(test['label'], score)
		return auc	
	elif cfg['lib'] == 'xgb':
		watch_list = [(d_train, 'train'), (d_test, 'test')]
		
		print(str(rf_No) + " Train...")
		bst = xgbt.train(params, d_train, num_round, watch_list, early_stopping_rounds = early_stopping)
		print(str(rf_No) + " Train Done!")
		bst.save_model(model_path + '_' + str(rf_No) + '.model')
		
		features = get_feature(cfg)
		fmap_path =  None
		if fmap_path:
			create_feature_map(fmap_path, features)
			feature_importance = bst.get_score(fmap=fmap_path, importance_type='gain')
			sorted_feature_importance = sorted(feature_importance.items(), key=operator.itemgetter(1), reverse=True)
			sorted_feature_importance_path = os.path.join(cfg['sorted_feature_importance']) if 'sorted_feature_importance' in cfg else None
			if sorted_feature_importance_path:
				with open(sorted_feature_importance_path + '_' + str(rf_No) + '.csv', 'w', encoding=encoding) as f:
					f.write('feature_name,f_score\n')
					for feature_name, f_score in sorted_feature_importance:
						f.write('{0},{1}\n'.format(feature_name, f_score))
		score = bst.predict(d_test)
		auc = roc_auc_score(test['label'], score)
		return auc


def work(cfg):
	rf_num = cfg['rf_num']
	data = read_data(cfg)
	model_auc_list = []
	if rf_num <= 1:
		d_train, d_test, test = prepare_data(data, 0, cfg)
		auc = train(d_train, d_test, 0, cfg, test)
		model_auc_list.append((0,auc))
	else:
		for i in range(rf_num):
			d_train, d_test,test = prepare_data(data, i, cfg)
			auc = train(d_train, d_test, i, cfg, test)
			model_auc_list.append((i,auc))
			if cfg['rf_test'] != 0:
				break	
	print('result:\n')
	for i in model_auc_list:
		print(i)

if __name__ == '__main__':
	if len(sys.argv) < 2:
		print_help()
		exit(1)

	path = sys.argv[1]
	cfg = None

	with open(path, 'r', encoding = encoding) as f_cfg:
		cfg = json.load(f_cfg)

	work(cfg)
	
