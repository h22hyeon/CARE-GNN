import argparse
import torch
import time
import numpy as np
from collections import defaultdict, OrderedDict
import json

from train import ModelHandler
from utils import set_seeds

def set_random_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)


def main(config):
	config_lines = print_config(config)
	set_seeds(config['seed'])
	model = ModelHandler(config)
	gnn_auc, gnn_recall, gnn_f1 = model.train()

def multi_run_main(config):
	config_lines = print_config(config)
	hyperparams = []
	for k, v in config.items():
		if isinstance(v, list): # Multiple run일 경우 해당 실험에 사용된 seed를 기록한다.
			hyperparams.append(k)

	f1_list, auc_list, recall_list = [], [], []
	# configuration 오브젝트들을 튜플로 저장한다.
	configs = grid(config)
	for i, cnf in enumerate(configs):
		print('Running {}:\n'.format(i))
		# print(cnf['save_dir'])
		set_random_seed(cnf['seed'])
		st = time.time()
		model = ModelHandler(cnf, ckp)
		# AUC-ROC / Recall / F1-macro를 기록한다.
		gnn_auc, gnn_recall, gnn_f1 = model.train()
		f1_list.append(gnn_f1)
		auc_list.append(gnn_auc)
		recall_list.append(gnn_recall)
		print("Running {} done, elapsed time {}s".format(i, time.time()-st))


	# 기록된 AUC-ROC / Gmean의 평균을 계산하도록 한다.
	f1_mean, f1_std = np.mean(f1_list), np.std(f1_list, ddof=1)
	auc_mean, auc_std = np.mean(auc_list), np.std(auc_list, ddof=1)
	recall_mean, recall_std = np.mean(recall_list), np.std(recall_list, ddof=1)

	
	
# def get_args():
# 	parser = argparse.ArgumentParser()
	# dataset and model dependent args
	# parser.add_argument('--multi_run', action='store_true', help='flag: multi run')
	# parser.add_argument('--data_name', type=str, default='yelp', help='The dataset name. [yelp, amazon]')
	# parser.add_argument('--data_dir', type=str, default='/data/graphs_v3', help='The dataset dir path')
	# parser.add_argument('--model', type=str, default='CARE', help='The model name. [CARE, SAGE]')
	# parser.add_argument('--inter', type=str, default='GNN', help='The inter-relation aggregator type. [Att, Weight, Mean, GNN]')
	# parser.add_argument('--seed', type=list, default=[0,1,2,3,4,5,6,7,8,9], action="append", help='number of samples for each layer')
	# parser.add_argument('--batch_size', type=int, default=1024, help='Batch size 1024 for yelp, 256 for amazon.')
	# parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
	# parser.add_argument('--lambda_1', type=float, default=2, help='Simi loss weight.')
	# parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay (L2 loss weight).')
	# parser.add_argument('--emb_size', type=int, default=64, help='Node embedding size at the last layer.')
	# parser.add_argument('--num_epochs', type=int, default=101, help='Number of epochs.')
	# parser.add_argument('--valid_epochs', type=int, default=2, help='Number of epochs.')
	# parser.add_argument('--step_size', type=float, default=2e-2, help='RL action step size')
	# parser.add_argument('--graph_id', type=int, default=0, help='random seed')
	# parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
	# parser.add_argument('--cuda_id', type=int, default=0, help='GPU index')
	# args = vars(parser.parse_args())
	# return args

def get_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('--exp_config_path', type=str, default='./experiment_configs/template_GCN.json')
	args = vars(parser.parse_args())
	return args

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

def grid(kwargs):
	"""Builds a mesh grid with given keyword arguments for this Config class.
	If the value is not a list, then it is considered fixed"""

	class MncDc:
		"""This is because np.meshgrid does not always work properly..."""

		def __init__(self, a):
			self.a = a  # tuple!

		def __call__(self):
			return self.a

	def merge_dicts(*dicts):
		"""
		Merges dictionaries recursively. Accepts also `None` and returns always a (possibly empty) dictionary
		"""
		from functools import reduce
		def merge_two_dicts(x, y):
			z = x.copy()  # start with x's keys and values
			z.update(y)  # modifies z with y's keys and values & returns None
			return z

		return reduce(lambda a, nd: merge_two_dicts(a, nd if nd else {}), dicts, {})


	sin = OrderedDict({k: v for k, v in kwargs.items() if isinstance(v, list)})
	for k, v in sin.items():
		copy_v = []
		for e in v:
			copy_v.append(MncDc(e) if isinstance(e, tuple) else e)
		sin[k] = copy_v

	grd = np.array(np.meshgrid(*sin.values()), dtype=object).T.reshape(-1, len(sin.values()))
	return [merge_dicts(
		{k: v for k, v in kwargs.items() if not isinstance(v, list)},
		{k: vv[i]() if isinstance(vv[i], MncDc) else vv[i] for i, k in enumerate(sin)}
	) for vv in grd]


################################################################################
# Module Command-line Behavior #
################################################################################
if __name__ == '__main__':
	args = get_arguments()
	with open(args['exp_config_path']) as f:
		args = json.load(f)
	main(args)
