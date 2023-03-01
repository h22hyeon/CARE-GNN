import pickle
import random as rd
import numpy as np
import scipy.sparse
import scipy.sparse as sp
from scipy.io import loadmat
import copy as cp
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, average_precision_score
from collections import defaultdict
from datetime import datetime
import os



"""
	Utility functions to handle data and evaluate model.
"""
class log:
	def __init__(self, model_name=None):
		self.time_step = str(datetime.now())
		self.save_dir_path = f"/data/Save_model({model_name})"
		self.log_dir_path = "./log"
		self.log_file_name = f"({model_name})" + self.time_step + ".log"
		self.train_log_path = os.path.join(self.log_dir_path, "train", self.log_file_name)
		self.valid_log_path = os.path.join(self.log_dir_path, "valid", self.log_file_name)
		self.test_log_path = os.path.join(self.log_dir_path, "test", self.log_file_name)
		self.multi_run_log_path = os.path.join(self.log_dir_path, "multi-run(total)", self.log_file_name)
		os.makedirs(os.path.join(self.log_dir_path, "train"), exist_ok=True)
		os.makedirs(os.path.join(self.log_dir_path, "valid"), exist_ok=True)
		os.makedirs(os.path.join(self.log_dir_path, "test"), exist_ok=True)
		os.makedirs(os.path.join(self.log_dir_path, "multiple-run"), exist_ok=True)

	def write_train_log(self, line, print_line=True):
		if print_line:
			print(line)
		log_file = open(self.train_log_path, 'a')
		log_file.write(line + "\n")
		log_file.close()

	def write_valid_log(self, line, print_line=True):
		if print_line:
			print(line)
		log_file = open(self.valid_log_path, 'a')
		log_file.write(line + "\n")
		log_file.close()

	def write_test_log(self, line, print_line=True):
		if print_line:
			print(line)
		log_file = open(self.test_log_path, 'a')
		log_file.write(line + "\n")
		log_file.close()
	
	def multi_run_log(self, line, print_line=True):
		if print_line:
			print(line)
		log_file = open(self.multi_run_log_path, 'a')
		log_file.write(line + "\n")
		log_file.close()

def print_config(config):
    print("**************** MODEL CONFIGURATION ****************")
    # Configuration 파일을 불러와 train setting을 출력한다.
    config_lines = ""
    for key in sorted(config.keys()):
        val = config[key]
        keystr = "{}".format(key) + (" " * (24 - len(key)))
        line = "{} -->   {}\n".format(keystr, val)
        config_lines += line
        print(line)
    print("**************** MODEL CONFIGURATION ****************")
    
    return config_lines

def load_data(data, prefix = 'data/', graph_id=None):
	"""
	Load graph, feature, and label given dataset name
	:returns: home and single-relation graphs, feature, label
	"""
	if data == 'yelp':
		data_file = loadmat(prefix + 'YelpChi.mat')
		labels = data_file['label'].flatten()
		feat_data = data_file['features'].todense().A
		# load the preprocessed adj_lists
		with open(prefix + 'yelp_homo_adjlists.pickle', 'rb') as file:
			homo = pickle.load(file)
		file.close()
		with open(prefix + 'yelp_rur_adjlists.pickle', 'rb') as file:
			relation1 = pickle.load(file)
		file.close()
		with open(prefix + 'yelp_rtr_adjlists.pickle', 'rb') as file:
			relation2 = pickle.load(file)
		file.close()
		with open(prefix + 'yelp_rsr_adjlists.pickle', 'rb') as file:
			relation3 = pickle.load(file)
		file.close()
		relation_list = [relation1, relation2, relation3]

	elif data == 'amazon':
		data_file = loadmat(prefix + 'Amazon.mat')
		labels = data_file['label'].flatten()
		feat_data = data_file['features'].todense().A
		# load the preprocessed adj_lists
		with open(prefix + 'amz_homo_adjlists.pickle', 'rb') as file:
			homo = pickle.load(file)
		file.close()
		with open(prefix + 'amz_upu_adjlists.pickle', 'rb') as file:
			relation1 = pickle.load(file)
		file.close()
		with open(prefix + 'amz_usu_adjlists.pickle', 'rb') as file:
			relation2 = pickle.load(file)
		file.close()
		with open(prefix + 'amz_uvu_adjlists.pickle', 'rb') as file:
			relation3 = pickle.load(file)
		file.close()
		relation_list = [relation1, relation2, relation3]
	elif data == 'KDK':
		postfix = "(CSC).npz"

		graph_num = str(graph_id).zfill(3)
		feature_path = os.path.join(prefix, "attributes", graph_num + "_node_feature" + postfix)
		label_path = os.path.join(prefix, "labels", graph_num + "_label.npy")

		labels = np.load(label_path).flatten()
		feat_data = scipy.sparse.load_npz(feature_path).astype(float).todense().A

		network_dir_path_hetero = os.path.join(prefix, "G0_Hetero")
		network_type_list = ["_c_acc_c_network", "_c_clcare_c_network", "_c_fp_c_network",
						 "_c_hsdrcare_c_network","_c_insr_c_network"]
		network_path_list = [os.path.join(network_dir_path_hetero, graph_num + network_type_list[i] + postfix) for i in range(len(network_type_list))]
		relation_list = [scipy.sparse.load_npz(network_path_list[i]) for i in range(len(network_path_list))]
		for i, relation in enumerate(relation_list):
			relation_list[i] = sparse_to_adjlist_for_train(relation + sp.eye(relation.shape[0]))
		
		network_dir_path_homo = os.path.join(prefix, "G0_Homo")
		homo_network_path = os.path.join(network_dir_path_homo, graph_num + "_G0_Homo_network" + postfix)
		homo = scipy.sparse.load_npz(homo_network_path)
		homo = sparse_to_adjlist_for_train(homo + sp.eye(homo.shape[0]))

	return homo, relation_list, feat_data, labels


def normalize(mx):
	"""
		Row-normalize sparse matrix
		Code from https://github.com/williamleif/graphsage-simple/
	"""
	rowsum = np.array(mx.sum(1)) + 0.01
	r_inv = np.power(rowsum, -1).flatten()
	r_inv[np.isinf(r_inv)] = 0.
	r_mat_inv = sp.diags(r_inv)
	mx = r_mat_inv.dot(mx)
	return mx


def sparse_to_adjlist(sp_matrix, filename):
	"""
	Transfer sparse matrix to adjacency list
	:param sp_matrix: the sparse matrix
	:param filename: the filename of adjlist
	"""
	# add self loop
	homo_adj = sp_matrix + sp.eye(sp_matrix.shape[0])
	# create adj_list
	adj_lists = defaultdict(set)
	edges = homo_adj.nonzero()
	for index, node in enumerate(edges[0]):
		adj_lists[node].add(edges[1][index])
		adj_lists[edges[1][index]].add(node)
	with open(filename, 'wb') as file:
		pickle.dump(adj_lists, file)
	file.close()

def sparse_to_adjlist_for_train(sp_matrix): # CSC 포멧의 인접행렬을 adjlist 형태로 변환하는 함수로 학습 과정에서 사용한다.
	# add self loop
	homo_adj = sp_matrix + sp.eye(sp_matrix.shape[0]) 
	# create adj_list
	adj_lists = defaultdict(set)
	edges = homo_adj.nonzero() # 해당 함수는 non-zero value를 갖는 인덱스 (row, col)을 리스트 퓨플로 반환한다.
	for index, node in enumerate(edges[0]): 
		adj_lists[node].add(edges[1][index])
		adj_lists[edges[1][index]].add(node) # symmetric relation을 커버하기 위한 코드.
	
	return adj_lists


def pos_neg_split(nodes, labels):
	"""
	Find positive and negative nodes given a list of nodes and their labels
	:param nodes: a list of nodes
	:param labels: a list of node labels
	:returns: the spited positive and negative nodes
	"""
	pos_nodes = []
	neg_nodes = cp.deepcopy(nodes)
	aux_nodes = cp.deepcopy(nodes)
	for idx, label in enumerate(labels):
		if label == 1:
			pos_nodes.append(aux_nodes[idx])
			neg_nodes.remove(aux_nodes[idx])

	return pos_nodes, neg_nodes


def undersample(pos_nodes, neg_nodes, scale=1):
	"""
	Under-sample the negative nodes
	:param pos_nodes: a list of positive nodes
	:param neg_nodes: a list negative nodes
	:param scale: the under-sampling scale
	:return: a list of under-sampled batch nodes
	"""

	aux_nodes = cp.deepcopy(neg_nodes)
	aux_nodes = rd.sample(aux_nodes, k=int(len(pos_nodes)*scale))
	batch_nodes = pos_nodes + aux_nodes

	return batch_nodes


def test_sage(test_cases, labels, model, batch_size, ckp, flag=None):
	"""
	Test the performance of GraphSAGE
	:param test_cases: a list of testing node
	:param labels: a list of testing node labels
	:param model: the GNN model
	:param batch_size: number nodes in a batch
	"""

	test_batch_num = int(len(test_cases) / batch_size) + 1
	f1_gnn = 0.0
	acc_gnn = 0.0
	recall_gnn = 0.0
	gnn_list = []
	for iteration in range(test_batch_num):
		i_start = iteration * batch_size
		i_end = min((iteration + 1) * batch_size, len(test_cases))
		batch_nodes = test_cases[i_start:i_end]
		batch_label = labels[i_start:i_end]
		gnn_prob = model.to_prob(batch_nodes)
		f1_gnn += f1_score(batch_label, gnn_prob.data.cpu().numpy().argmax(axis=1), average="macro")
		acc_gnn += accuracy_score(batch_label, gnn_prob.data.cpu().numpy().argmax(axis=1))
		recall_gnn += recall_score(batch_label, gnn_prob.data.cpu().numpy().argmax(axis=1), average="macro")
		gnn_list.extend(gnn_prob.data.cpu().numpy()[:, 1].tolist())

	auc_gnn = roc_auc_score(labels, np.array(gnn_list))
	ap_gnn = average_precision_score(labels, np.array(gnn_list))
	line1= f"GNN F1: {f1_gnn / test_batch_num:.4f}|\tGNN Accuracy: {acc_gnn / test_batch_num:.4f}|"+\
       f"\tGNN Recall: {recall_gnn / test_batch_num:.4f}|\tGNN auc: {auc_gnn:.4f}|\tGNN ap: {ap_gnn:.4f}"
		
	if flag=="val":
		ckp.write_valid_log("Validation: "+ line1)
	elif flag=="test":
		ckp.write_test_log("Test: "+ line1)
	return auc_gnn,(recall_gnn / test_batch_num), (f1_gnn / test_batch_num)


def test_care(test_cases, labels, model, batch_size, ckp, flag=None):
	"""
	Test the performance of CARE-GNN and its variants
	:param test_cases: a list of testing node
	:param labels: a list of testing node labels
	:param model: the GNN model
	:param batch_size: number nodes in a batch
	:returns: the AUC and Recall of GNN and Simi modules
	"""

	test_batch_num = int(len(test_cases) / batch_size) + 1
	f1_gnn = 0.0
	acc_gnn = 0.0
	recall_gnn = 0.0
	f1_label1 = 0.0
	acc_label1 = 0.00
	recall_label1 = 0.0
	gnn_list = []
	label_list1 = []

	for iteration in range(test_batch_num):
		i_start = iteration * batch_size
		i_end = min((iteration + 1) * batch_size, len(test_cases))
		batch_nodes = test_cases[i_start:i_end]
		batch_label = labels[i_start:i_end]

		# 학습된 CARE-GNN 모델을 통해 반환되는 GNN score와 label-aware score를 통해 성능 평가를 수행한다.
		gnn_prob, label_prob1 = model.to_prob(batch_nodes, batch_label, train_flag=False)

		f1_gnn += f1_score(batch_label, gnn_prob.data.cpu().numpy().argmax(axis=1), average="macro")
		acc_gnn += accuracy_score(batch_label, gnn_prob.data.cpu().numpy().argmax(axis=1))
		recall_gnn += recall_score(batch_label, gnn_prob.data.cpu().numpy().argmax(axis=1), average="macro")

		f1_label1 += f1_score(batch_label, label_prob1.data.cpu().numpy().argmax(axis=1), average="macro")
		acc_label1 += accuracy_score(batch_label, label_prob1.data.cpu().numpy().argmax(axis=1))
		recall_label1 += recall_score(batch_label, label_prob1.data.cpu().numpy().argmax(axis=1), average="macro")

		gnn_list.extend(gnn_prob.data.cpu().numpy()[:, 1].tolist())
		label_list1.extend(label_prob1.data.cpu().numpy()[:, 1].tolist())

	auc_gnn = roc_auc_score(labels, np.array(gnn_list))
	ap_gnn = average_precision_score(labels, np.array(gnn_list))
	auc_label1 = roc_auc_score(labels, np.array(label_list1))
	ap_label1 = average_precision_score(labels, np.array(label_list1))
	
# 	line1= f"GNN F1: {f1_gnn / test_batch_num:.4f}|\tGNN Accuracy: {acc_gnn / test_batch_num:.4f}|"+\
#        f"\tGNN Recall: {recall_gnn / test_batch_num:.4f}|\tGNN auc: {auc_gnn:.4f}|\tGNN ap: {ap_gnn:.4f}"
# 	line2 = f"Label1 F1: {f1_label1 / test_batch_num:.4f}|\tLabel1 Accuracy: {acc_label1 / test_batch_num:.4f}|"+\
#        f"\tLabel1 Recall: {recall_label1 / test_batch_num:.4f}|\tLabel1 auc: {auc_label1:.4f}|\tLabel1 ap: {ap_label1:.4f}"
	
	line1= f"GNN F1: {f1_gnn / test_batch_num:.4f}\tGNN Accuracy: {acc_gnn / test_batch_num:.4f}"+\
       f"\tGNN Recall: {recall_gnn / test_batch_num:.4f}\tGNN AUC-ROC: {auc_gnn:.4f}\tGNN AP: {ap_gnn:.4f}"

	if flag=="val":
		ckp.write_valid_log("Validation: "+ line1)
		# ckp.write_valid_log("Validation: "+ line2, print_line=False)
	elif flag=="test":
		ckp.write_test_log("Test: "+ line1)
		# ckp.write_test_log("Test: "+ line2, print_line=False)

	return auc_gnn, recall_gnn, (f1_gnn / test_batch_num)