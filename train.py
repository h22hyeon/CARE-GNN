import time
import os
import random
import argparse
from sklearn.model_selection import train_test_split

from utils import *
from model import *
from layers import *
from graphsage import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

"""
	Training CARE-GNN
	Paper: Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters
	Source: https://github.com/YingtongDou/CARE-GNN
"""

parser = argparse.ArgumentParser()

# dataset and model dependent args
parser.add_argument('--data_name', type=str, default='KDK', help='The dataset name. [yelp, amazon]')
parser.add_argument('--data_dir', type=str, default='/data/graphs_v3', help='The dataset dir path')
parser.add_argument('--model', type=str, default='CARE', help='The model name. [CARE, SAGE]')
parser.add_argument('--inter', type=str, default='GNN', help='The inter-relation aggregator type. [Att, Weight, Mean, GNN]')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--batch-size', type=int, default=1024, help='Batch size 1024 for yelp, 256 for amazon.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--lambda_1', type=float, default=2, help='Simi loss weight.')
parser.add_argument('--lambda_2', type=float, default=1e-3, help='Weight decay (L2 loss weight).')
parser.add_argument('--emb_size', type=int, default=64, help='Node embedding size at the last layer.')
parser.add_argument('--num-epochs', type=int, default=101, help='Number of epochs.')
parser.add_argument('--valid_epochs', type=int, default=10, help='Number of epochs.')
parser.add_argument('--under-sample', type=int, default=1, help='Under-sampling scale.')
parser.add_argument('--step-size', type=float, default=2e-2, help='RL action step size')
parser.add_argument('--graph_id', type=int, default=0, help='random seed')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--cuda_id', type=int, default=0, help='GPU index')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device(f'cuda:{args.cuda_id}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)
print(f'run on {args.data_name}')

# load graph, feature, and label (relation list로 여러 타입의 relation을 받아오도록 변경함.)
homo, relation_list, feat_data, labels = load_data(args.data_name, prefix=args.data_dir, graph_id=args.graph_id)

# train_test split
np.random.seed(args.seed)
random.seed(args.seed)

ckp = log(args.model)
config_lines = print_config(vars(args))
ckp.write_train_log(config_lines, print_line=False)
ckp.write_valid_log(config_lines, print_line=False)
ckp.write_test_log(config_lines, print_line=False)

# Label의 비율을 동일하게 가져가기 위해 계층적 샘플링을 수행한다.
if args.data_name == 'yelp' or args.data_name == 'KDK':
	index = list(range(len(labels)))
	idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels, stratify=labels, train_size=0.4,
																	random_state=2, shuffle=True)
	idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest, test_size=0.67,
																	random_state=2, shuffle=True)

elif args.data_name == 'amazon':  # amazon
	# 0-3304 are unlabeled nodes - Amazon 데이터 셋은 unlabeld 노드가 존재한다.
	index = list(range(3305, len(labels)))
	idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels[3305:], stratify=labels[3305:],
																	train_size=0.4, random_state=2, shuffle=True)
	idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
																	test_size=0.67, random_state=2, shuffle=True)

print(f'Run on {args.data_name}, postive/total num: {np.sum(labels)}/{len(labels)}, train num {len(y_train)},'+
			f'valid num {len(y_valid)}, test num {len(y_test)}, test positive num {np.sum(y_test)}')

# split pos neg sets for under-sampling
train_pos, train_neg = pos_neg_split(idx_train, y_train) # Training set에서 positive/negative sample에 대한 인덱스를 구분한다.

# initialize model input
features = nn.Embedding(feat_data.shape[0], feat_data.shape[1])
feat_data = normalize(feat_data) # feature를 Row-normalize sparse matrix로 변환한다. 
features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False) # feature를 임베딩한다 (고정).

if args.cuda:
	features.cuda()

# set input graph
if args.model == 'SAGE':
	adj_lists = homo
else:
	adj_lists = relation_list

print(f'Model: {args.model}, Inter-AGG: {args.inter}, emb_size: {args.emb_size}.')

# build one-layer models
if args.model == 'CARE' and args.data_name=='KDK':
	intra1 = IntraAgg(features, feat_data.shape[1], cuda=args.cuda)
	intra2 = IntraAgg(features, feat_data.shape[1], cuda=args.cuda)
	intra3 = IntraAgg(features, feat_data.shape[1], cuda=args.cuda)
	intra4 = IntraAgg(features, feat_data.shape[1], cuda=args.cuda)
	intra5 = IntraAgg(features, feat_data.shape[1], cuda=args.cuda)
	inter1 = InterAgg5(features, feat_data.shape[1], args.emb_size, adj_lists, [intra1, intra2, intra3, intra4, intra5], inter=args.inter,
					  step_size=args.step_size, cuda=args.cuda)
	
elif args.model == 'CARE' and (args.data_name=='yelp' or args.data_name == 'amazon'):
	intra1 = IntraAgg(features, feat_data.shape[1], cuda=args.cuda)
	intra2 = IntraAgg(features, feat_data.shape[1], cuda=args.cuda)
	intra3 = IntraAgg(features, feat_data.shape[1], cuda=args.cuda)
	inter1 = InterAgg3(features, feat_data.shape[1], args.emb_size, adj_lists, [intra1, intra2, intra3], inter=args.inter,
					  step_size=args.step_size, cuda=args.cuda)
elif args.model == 'SAGE':
	agg1 = MeanAggregator(features, cuda=args.cuda)
	enc1 = Encoder(features, feat_data.shape[1], args.emb_size, adj_lists, agg1, gcn=True, cuda=args.cuda)

if args.model == 'CARE':
	gnn_model = OneLayerCARE(2, inter1, args.lambda_1)
elif args.model == 'SAGE':
	# the vanilla GraphSAGE model as baseline
	enc1.num_samples = 5
	gnn_model = GraphSage(2, enc1)

if args.cuda:
	gnn_model.cuda()

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gnn_model.parameters()), lr=args.lr, weight_decay=args.lambda_2)
times = []

dir_saver = os.path.join("/data/CARE-GNN_models/", ckp.log_file_name)
os.makedirs(dir_saver,exist_ok=True)
path_saver = os.path.join(dir_saver, '{}_{}.pkl'.format(args.data_name, args.model))
f1_mac_best, auc_best, ep_best = 0, 0, -1

# train the model
for epoch in range(args.num_epochs):
	# randomly under-sampling negative nodes for each epoch
	sampled_idx_train = undersample(train_pos, train_neg, scale=1)
	rd.shuffle(sampled_idx_train)

	# send number of batches to model to let the RLModule know the training progress
	num_batches = int(len(sampled_idx_train) / args.batch_size) + 1
	if args.model == 'CARE':
		inter1.batch_num = num_batches

	loss = 0.0
	epoch_time = 0

	# mini-batch training
	"""
	자세한 학습 알고리즘은 layers.py를 통해 확인할 수 있다. 
	"""
	for batch in range(num_batches):
		start_time = time.time()
		i_start = batch * args.batch_size
		i_end = min((batch + 1) * args.batch_size, len(sampled_idx_train))
		batch_nodes = sampled_idx_train[i_start:i_end]
		batch_label = labels[np.array(batch_nodes)]
		optimizer.zero_grad()
		
		if args.cuda:
			loss = gnn_model.loss(batch_nodes, Variable(torch.cuda.LongTensor(batch_label)))
		else:
			loss = gnn_model.loss(batch_nodes, Variable(torch.LongTensor(batch_label)))
		loss.backward()
		optimizer.step()
		end_time = time.time()
		epoch_time += end_time - start_time
		loss += loss.item()

	print(f'Epoch: {epoch}, loss: {loss.item() / num_batches}, time: {epoch_time}s')

	"""
	Test 과정에 대한 알고리즘은 utils.py에서 확인할 수 있다.
	"""
	# Valid the model for every $valid_epoch$ epoch
	if (epoch+1) % args.valid_epochs == 0:
		if args.model == 'SAGE' or args.model == 'GCN':
			print("Valid at epoch {}".format(epoch))
			gnn_auc_val, gnn_recall_val, gnn_f1_val = test_sage(idx_test, y_test, gnn_model, args.batch_size, ckp, flag="val")
			if gnn_auc_val > auc_best:
				gnn_recall_best, f1_mac_best, auc_best, ep_best = gnn_recall_val, gnn_f1_val, gnn_auc_val, epoch
				if not os.path.exists(dir_saver):
					os.makedirs(dir_saver)
				print('  Saving model ...')
				torch.save(gnn_model.state_dict(), path_saver)
		# CARE-GNN을 학습할 경우의 test!
		else:
			print("Valid at epoch {}".format(epoch))
			gnn_auc_val, gnn_recall_val, gnn_f1_val = test_care(idx_test, y_test, gnn_model, args.batch_size, ckp, flag="val")
			if gnn_auc_val > auc_best:
				gnn_recall_best, f1_mac_best, auc_best, ep_best = gnn_recall_val, gnn_f1_val, gnn_auc_val, epoch
				if not os.path.exists(dir_saver):
					os.makedirs(dir_saver)
				print('  Saving model ...')
				torch.save(gnn_model.state_dict(), path_saver)

if args.model == 'SAGE':
	gnn_auc, gnn_recall, gnn_f1 = test_sage(idx_test, y_test, gnn_model, args.batch_size, ckp, flag="test")
else:
	gnn_auc, gnn_recall, gnn_f1 = test_care(idx_test, y_test, gnn_model, args.batch_size, ckp, flag="test")