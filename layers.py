import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable


from operator import itemgetter
import math

"""
	CARE-GNN Layers
	Paper: Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters
	Source: https://github.com/YingtongDou/CARE-GNN
"""


class InterAgg(nn.Module):

	def __init__(self, features, feature_dim,
				 embed_dim, adj_lists, intraggs,
				 inter='GNN', step_size=0.02, cuda=True):
		"""
		Initialize the inter-relation aggregator
		:param features: the input node features or embeddings for all nodes
		:param feature_dim: the input dimension
		:param embed_dim: the output dimension
		:param adj_lists: a list of adjacency lists for each single-relation graph
		:param intraggs: the intra-relation aggregators used by each single-relation graph
		:param inter: the aggregator type: 'Att', 'Weight', 'Mean', 'GNN'
		:param step_size: the RL action step size
		:param cuda: whether to use GPU
		"""
		super(InterAgg, self).__init__()

		self.features = features
		self.dropout = 0.6
		self.adj_lists = adj_lists
		self.intra_agg1 = intraggs[0]
		self.intra_agg2 = intraggs[1]
		self.intra_agg3 = intraggs[2]
		self.embed_dim = embed_dim
		self.feat_dim = feature_dim
		self.inter = inter
		self.step_size = step_size
		self.cuda = cuda
		self.intra_agg1.cuda = cuda
		self.intra_agg2.cuda = cuda
		self.intra_agg3.cuda = cuda

		# RL condition flag
		self.RL = True

		# number of batches for current epoch, assigned during training
		self.batch_num = 0

		# initial filtering thresholds
		self.thresholds = [0.5, 0.5, 0.5]

		# the activation function used by attention mechanism
		self.leakyrelu = nn.LeakyReLU(0.2)

		# parameter used to transform node embeddings before inter-relation aggregation
		self.weight = nn.Parameter(torch.FloatTensor(self.feat_dim, self.embed_dim))
		init.xavier_uniform_(self.weight)

		# weight parameter for each relation used by CARE-Weight
		self.alpha = nn.Parameter(torch.FloatTensor(self.embed_dim, 3))
		init.xavier_uniform_(self.alpha)

		# parameters used by attention layer
		self.a = nn.Parameter(torch.FloatTensor(2 * self.embed_dim, 1))
		init.xavier_uniform_(self.a)

		# label predictor for similarity measure
		self.label_clf = nn.Linear(self.feat_dim, 2)

		# initialize the parameter logs
		self.weights_log = []
		self.thresholds_log = [self.thresholds]
		self.relation_score_log = []

	def forward(self, nodes, labels, train_flag=True):
		"""
		:param nodes: a list of batch node ids
		:param labels: a list of batch node labels, only used by the RLModule
		:param train_flag: indicates whether in training or testing mode
		:return combined: the embeddings of a batch of input node features
		:return center_scores: the label-aware scores of batch nodes
		"""

		# extract 1-hop neighbor ids from adj lists of each single-relation graph
		to_neighs = [] # 각 component에는 각 relation에 따른 타겟 노드의 이웃 노드의 인덱스가 존재함.
		for adj_list in self.adj_lists: # 각 relation 별로
			to_neighs.append([set(adj_list[int(node)]) for node in nodes]) # 배치를 구성하는 노드에 대한 이웃 노드를 추출한다.

		# find unique nodes and their neighbors used in current batch
		unique_nodes = set.union(set.union(*to_neighs[0]), set.union(*to_neighs[1]), # 배치에서 사용되는 모든 노드(이웃 노드들 포함)의 집합을 생성한다.
								 set.union(*to_neighs[2], set(nodes)))

		# calculate label-aware scores
		if self.cuda:
			# 배치에서 사용되는 모든 노드들에 대한 피처를 batch_features로 정의한다.
			batch_features = self.features(torch.cuda.LongTensor(list(unique_nodes)))
		else:
			batch_features = self.features(torch.LongTensor(list(unique_nodes)))
		# batch_features를 통해 생성한 label aware score를 batch_scores로 정의한다.
		batch_scores = self.label_clf(batch_features)
		
		# 각 노드에 대한 매핑 딕셔너리를 생성한다.
		# key: 전체 그래프에 대한 해당 노드의 인덱스
		# value: unique_nodes set을 list로 변환했을 떄 해당 노드의 인덱스
		"""
		아래의 과정을 통해 딕셔너리는 node_id를 기준으로 unique_nodes에 존재하는 인덱스를 가리키게 된다.
		batch_scores에 존재하는 값이 unique_nodes에 존재하는 노드 인덱스와 대응되므로 score를 가져올 때는 id_mapping를 이용한다. 
		"""
		id_mapping = {node_id: index for node_id, index in zip(unique_nodes, range(len(unique_nodes)))}

		# the label-aware scores for current batch of nodes
		# 배치를 구성하는 노드(타겟 노드)들의 similarity score를 center_scores로 정의한다.
		center_scores = batch_scores[itemgetter(*nodes)(id_mapping), :]

		# get neighbor node id list for each batch node and relation
		# 각 relation에 따른 타겟 노드의 이웃 노드들의 리스트로 하여 r1_list를 생성한다 (이중 리스트). 
		r1_list = [list(to_neigh) for to_neigh in to_neighs[0]]
		r2_list = [list(to_neigh) for to_neigh in to_neighs[1]]
		r3_list = [list(to_neigh) for to_neigh in to_neighs[2]]

		# assign label-aware scores to neighbor nodes for each batch node and relation
		# 각 relation에 따른 타겟 노드의 이웃 노드들의 similarity score를 리스트로 하여 r1_scores을 생성한다 (이중 리스트 내부에 tensor).
		r1_scores = [batch_scores[itemgetter(*to_neigh)(id_mapping), :].view(-1, 2) for to_neigh in r1_list]
		r2_scores = [batch_scores[itemgetter(*to_neigh)(id_mapping), :].view(-1, 2) for to_neigh in r2_list]
		r3_scores = [batch_scores[itemgetter(*to_neigh)(id_mapping), :].view(-1, 2) for to_neigh in r3_list]

		# count the number of neighbors kept for aggregation for each batch node and relation
		# 각 relation에 따른 타겟 노드의 이웃 선택 비율을 threshold를 통해 조정한다.
		"""
		이때 해당 threshold(self.thresholds)는 RL 모듈을 통해 학습된다.
		"""
		r1_sample_num_list = [math.ceil(len(neighs) * self.thresholds[0]) for neighs in r1_list]
		r2_sample_num_list = [math.ceil(len(neighs) * self.thresholds[1]) for neighs in r2_list]
		r3_sample_num_list = [math.ceil(len(neighs) * self.thresholds[2]) for neighs in r3_list]

		# intra-aggregation steps for each relation
		# Eq. (8) in the paper
		# 배치를 구성하는 타겟 노드들의 이웃 노드의 인덱스와 similarity secore를 통해 intra aggregation을 수행한다.
		# 이때 최종적으로 반환되는 것은 배치에 대한 각 relation별 embedding과 similarity score(RL 모듈 학습과 어떻게 연관 되는지 확인 필요)이다.  
		r1_feats, r1_scores = self.intra_agg1.forward(nodes, r1_list, center_scores, r1_scores, r1_sample_num_list)
		r2_feats, r2_scores = self.intra_agg2.forward(nodes, r2_list, center_scores, r2_scores, r2_sample_num_list)
		r3_feats, r3_scores = self.intra_agg3.forward(nodes, r3_list, center_scores, r3_scores, r3_sample_num_list)

		# concat the intra-aggregated embeddings from each relation
		# 각 relation을 이용해 생성된 embedding을 concatenation하여 neigh_feats를 정의한다.
		neigh_feats = torch.cat((r1_feats, r2_feats, r3_feats), dim=0)

		# get features or embeddings for batch nodes
		if self.cuda and isinstance(nodes, list):
			index = torch.LongTensor(nodes).cuda()
		else:
			index = torch.LongTensor(nodes)
		self_feats = self.features(index)

		# number of nodes in a batch
		n = len(nodes)

		# inter-relation aggregation steps
		# Eq. (9) in the paper
		"""
		Aggregator를 선택하여 이용한다. 논문에서는 각각의 aggregator를 이용하여 실험을 진행하였다.
		"""
		if self.inter == 'Att':
			# 1) CARE-Att Inter-relation Aggregator
			combined, attention = att_inter_agg(len(self.adj_lists), self.leakyrelu, self_feats, neigh_feats, self.embed_dim,
												self.weight, self.a, n, self.dropout, self.training, self.cuda)
		elif self.inter == 'Weight':
			# 2) CARE-Weight Inter-relation Aggregator
			combined = weight_inter_agg(len(self.adj_lists), self_feats, neigh_feats, self.embed_dim, self.weight, self.alpha, n, self.cuda)
			gem_weights = F.softmax(torch.sum(self.alpha, dim=0), dim=0).tolist()
			if train_flag:
				print(f'Weights: {gem_weights}')
		elif self.inter == 'Mean':
			# 3) CARE-Mean Inter-relation Aggregator
			combined = mean_inter_agg(len(self.adj_lists), self_feats, neigh_feats, self.embed_dim, self.weight, n, self.cuda)
		elif self.inter == 'GNN':
			# 4) CARE-GNN Inter-relation Aggregator
			combined = threshold_inter_agg(len(self.adj_lists), self_feats, neigh_feats, self.embed_dim, self.weight, self.thresholds, n, self.cuda)

		# the reinforcement learning module
		if self.RL and train_flag: # 해당 RL 모듈은 각 relation에서 이용할 top-k 이웃 노드의 비율(self.thresholds)을 결정한다.
			relation_scores, rewards, thresholds, stop_flag = RLModule([r1_scores, r2_scores, r3_scores],
																	   self.relation_score_log, labels, self.thresholds,
																	   self.batch_num, self.step_size)
			self.thresholds = thresholds
			self.RL = stop_flag
			self.relation_score_log.append(relation_scores)
			self.thresholds_log.append(self.thresholds)

		return combined, center_scores # 통합된 embedding과 배치의 각 노드에 대한 label-aware score를 반환한다.


class IntraAgg(nn.Module):

	def __init__(self, features, feat_dim, cuda=False):
		"""
		Initialize the intra-relation aggregator
		:param features: the input node features or embeddings for all nodes
		:param feat_dim: the input dimension
		:param cuda: whether to use GPU
		"""
		super(IntraAgg, self).__init__()

		self.features = features
		self.cuda = cuda
		self.feat_dim = feat_dim

	def forward(self, nodes, to_neighs_list, batch_scores, neigh_scores, sample_list):
		"""
		Code partially from https://github.com/williamleif/graphsage-simple/
		:param | nodes: list of nodes in a batch
		:param | to_neighs_list: neighbor node id list for each batch node in one relation
		:param | batch_scores: the label-aware scores of batch nodes
		:param | neigh_scores: the label-aware scores 1-hop neighbors each batch node in one relation
		:param | sample_list: the number of neighbors kept for each batch node in one relation
		
		:return | to_feats: the aggregated embeddings of batch nodes neighbors in one relation
		:return | samp_scores: the average neighbor distances for each relation after filtering
		"""

		# filer neighbors under given relation
		# 배치를 구성하는 타겟 노드와 이웃 노드의 label-aware score, 이웃 노드의 인덱스 그리고 message를 이용할 노드의 수(relation에 따라 다름.).
		# filter_neighs_ada_threshold는 최종적으로 배치에 존재하는 개별 노드에 대한 선택된 이웃 노드의 인덱스와 그들의 score diff를 반환한다.
		samp_neighs, samp_scores = filter_neighs_ada_threshold(batch_scores, neigh_scores, to_neighs_list, sample_list)

		# find the unique nodes among batch nodes and the filtered neighbors
		unique_nodes_list = list(set.union(*samp_neighs)) # 선택된 노드에 동일한 인덱스의 노드가 존재할 수 있으므로 이를 제거한다 (추후 adj를 생성하여 feature를 이용하기 위함).
		# 이웃 노드들에 대해 개별적으로 접근해야 하므로 노드 인덱스에 대한 딕셔너리를 정의한다.
		unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}

		# intra-relation aggregation only with sampled neighbors
		mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes))) # aggregation을 위한 배치 단위의 인접 행렬을 정의한다.
		# 배치의 각 노드에 대한 이웃 노드들에 차례로 접근하여 인덱스를 부여한다.
		# (이때 모든 배치의 노드에 대해 하나의 리스트로 결과가 생성한다. -> column_indices)
		# 타겟 노드의 이웃 노드의 인덱스를 하나의 리스트로 통합하여 column_indices로 정의한다.
		column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
		# column_indices로에서 이웃 노드의 인덱스가 몇 번째 타겟 노드의 것인지를 지정하기 위한 row_indices를 생성한다.
		row_indices = [i for i in range(len(samp_neighs)) for _ in range(len(samp_neighs[i]))]
		# mask의 row는 타겟 노드의 인덱스를 의미하고, column은 이웃 노드의 인덱스를 의미한다.
		# 배치 단위에서의 adj라고 보면 될 것 같다. 
		mask[row_indices, column_indices] = 1
		if self.cuda:
			mask = mask.cuda()
		# 각 노드의 차수를 정의한다.
		num_neigh = mask.sum(1, keepdim=True)
		# Mean aggregator이다.
		mask = mask.div(num_neigh)
		if self.cuda:
			embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
		else:
			embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
		# 앞서 생성한 배치 단위의 adg를 통해 노드 피처를 aggrgation 한다.
		to_feats = mask.mm(embed_matrix)
		to_feats = F.relu(to_feats)
		return to_feats, samp_scores # 노드의 representation (가중치 없이 그냥 mean aggregation 됨.)과 각 타겟 노드에 대한 선택된 이웃 노드들의 score diff를 반환한다.


def RLModule(scores, scores_log, labels, thresholds, batch_num, step_size):
	"""
	The reinforcement learning module.
	It updates the neighbor filtering threshold for each relation based
	on the average neighbor distances between two consecutive epochs.
	:param |scores: the neighbor nodes label-aware scores for each relation
	:param |scores_log: a list stores the relation average distances for each batch
	:param |labels: the batch node labels used to select positive nodes
	:param |thresholds: the current neighbor filtering thresholds for each relation
	:param |batch_num: numbers batches in an epoch
	:param |step_size: the RL action step size
	:return | relation_scores: the relation average distances for current batch
	:return | rewards: the reward for given thresholds in current epoch
	:return | new_thresholds: the new filtering thresholds updated according to the rewards
	:return | stop_flag: the RL terminal condition flag
	"""
	# RLModule([r1_scores, r2_scores, r3_scores],self.relation_score_log,
	#    labels, self.thresholds,self.batch_num, self.step_size)

	relation_scores = []
	stop_flag = True

	# only compute the average neighbor distances for positive nodes
	pos_index = (labels == 1).nonzero().tolist()
	pos_index = [i[0] for i in pos_index] # 배치 내에서 positive sample의 인덱스를 찾고 pos_index로 정의한다.

	# compute average neighbor distances for each relation
	for score in scores:
		# 각 노드의 relation에 해당하는 label-aware score에 대하여
		# Positive sample의 인덱스() 통해 배치 노드 중에서 score diff를 기준으로 정렬한다.
		pos_scores = itemgetter(*pos_index)(score)

		# 배치에 존재하는 positive sample의 수를 카운팅하여 neigh_count로 정의한다.
		neigh_count = sum([1 if isinstance(i, float) else len(i) for i in pos_scores])
		# 배치에 존재하는 positive sample의 score diff를 pos_sum으로 정의한다.
		pos_sum = [i if isinstance(i, float) else sum(i) for i in pos_scores]
		# neigh_count와 pos_sum은 인덱스가 맞게 설정된다.
		
		# 해당 relation으로 연결된 이웃들의 정보를 통해 계산된 노드들의 label aware score 평균을 relation_scores에 append 한다.
		relation_scores.append(sum(pos_sum) / neigh_count)
	"""
	# do not call RL module within the epoch or within the first two epochs!!!!
	"""
	if len(scores_log) % batch_num != 0 or len(scores_log) < 2 * batch_num:
		rewards = [0, 0, 0]
		new_thresholds = thresholds
	else:
		# update thresholds according to average scores in last epoch
		# Eq.(5) in the paper
		previous_epoch_scores = [sum(s) / batch_num for s in zip(*scores_log[-2 * batch_num:-batch_num])] # 이전 배치의 label aware score 평균.
		current_epoch_scores = [sum(s) / batch_num for s in zip(*scores_log[-batch_num:])] # 현재 배치의 label aware score 평균.

		# compute reward for each relation and update the thresholds according to reward
		# Eq. (6) in the paper
		# 논문에 제시된 stochastic reward에 대한 부분으로 평균 label aware score의 향상에 따른 reward 정의
		rewards = [1 if previous_epoch_scores[i] - s >= 0 else -1 for i, s in enumerate(current_epoch_scores)]
		# 지정된 step size를 통해 action을 하여 relation별 threshold를 조정한다.
		new_thresholds = [thresholds[i] + step_size if r == 1 else thresholds[i] - step_size for i, r in enumerate(rewards)] 

		# avoid overflow
		new_thresholds = [0.999 if i > 1 else i for i in new_thresholds]
		new_thresholds = [0.001 if i < 0 else i for i in new_thresholds]

		# print(f'epoch scores: {current_epoch_scores}')
		# print(f'rewards: {rewards}')
		# print(f'thresholds: {new_thresholds}')

	# TODO: add terminal condition

	return relation_scores, rewards, new_thresholds, stop_flag # relation별 label aware score의 평균과 relation별 reward, ., ..


def filter_neighs_ada_threshold(center_scores, neigh_scores, neighs_list, sample_list):
	"""
	Filter neighbors according label predictor result with adaptive thresholds
	:param center_scores: the label-aware scores of batch nodes
	:param neigh_scores: the label-aware scores 1-hop neighbors each batch node in one relation
	:param neighs_list: neighbor node id list for each batch node in one relation
	:param sample_list: the number of neighbors kept for each batch node in one relation
	:return samp_neighs: the neighbor indices and neighbor simi scores
	:return samp_scores: the average neighbor distances for each relation after filtering
	"""

	samp_neighs = []
	samp_scores = []
	for idx, center_score in enumerate(center_scores): # 배치를 구성하는 각 타겟 노드의 label-aware score
		center_score = center_scores[idx][0] # 타겟 노드의 label-aware score [0]
		neigh_score = neigh_scores[idx][:, 0].view(-1, 1) # 타겟 노드의 이웃 노드들의 label-aware score [0]
		center_score = center_score.repeat(neigh_score.size()[0], 1) #  이웃 노드의 수 만큼 타겟 노드의 label-aware score를 확장한다.
		neighs_indices = neighs_list[idx] # 이웃 노드의 인덱스를 neighs_indices로 저장한다.
		num_sample = sample_list[idx] # 타겟 노드에 대하여 message를 이용할 이웃 노드의 수를 num_sample로 저장한다.

		# compute the L1-distance of batch nodes and their neighbors
		# Eq. (2) in paper
		# 각 타겟 노드에 대하여 이웃 노드와의 labe-aware score의 차이를 게산하고 score_diff로 정의 한다.
		score_diff = torch.abs(center_score - neigh_score).squeeze()
		sorted_scores, sorted_indices = torch.sort(score_diff, dim=0, descending=False) # 그 값을 기준으로 인덱스를 정렬한다 (오룸차순으로).
		selected_indices = sorted_indices.tolist() # 정렬된 인덱스를 selected_indices로 정의한다.

		# top-p sampling according to distance ranking and thresholds
		# Section 3.3.1 in paper
		# 
		if len(neigh_scores[idx]) > num_sample + 1:
			# 선택된 이웃 노드들의 인덱스를 selected_neighs로 정의한다
			selected_neighs = [neighs_indices[n] for n in selected_indices[:num_sample]]
			# 선택된 이웃 노드들과 타겟 노드와의 score 차이를  selected_score_diff로 정의한다.
			selected_scores = sorted_scores.tolist()[:num_sample]
		else:
			# 타겟 노드의 이웃 노드가 1개 혹은 singleton 노드일 경우
			selected_neighs = neighs_indices
			selected_scores = score_diff.tolist()
			if isinstance(selected_scores, float):
				selected_scores = [selected_scores]

		# 노드들의 선택된 neighbor와 그들의 score diff를 저장한다
		samp_neighs.append(set(selected_neighs))
		samp_scores.append(selected_scores)

	return samp_neighs, samp_scores # 배치에 존재하는 각 노드들의 이웃의 인덱스와 그들의 score diff를 반환한다.


def mean_inter_agg(num_relations, self_feats, neigh_feats, embed_dim, weight, n, cuda):
	"""
	Mean inter-relation aggregator
	:param num_relations: number of relations in the graph
	:param self_feats: batch nodes features or embeddings
	:param neigh_feats: intra-relation aggregated neighbor embeddings for each relation
	:param embed_dim: the dimension of output embedding
	:param weight: parameter used to transform node embeddings before inter-relation aggregation
	:param n: number of nodes in a batch
	:param cuda: whether use GPU
	:return: inter-relation aggregated node embeddings
	"""

	# transform batch node embedding and neighbor embedding in each relation with weight parameter
	center_h = torch.mm(self_feats, weight)
	neigh_h = torch.mm(neigh_feats, weight)

	# initialize the final neighbor embedding
	if cuda:
		aggregated = torch.zeros(size=(n, embed_dim)).cuda()
	else:
		aggregated = torch.zeros(size=(n, embed_dim))

	# sum neighbor embeddings together
	for r in range(num_relations):
		aggregated += neigh_h[r * n:(r + 1) * n, :]

	# sum aggregated neighbor embedding and batch node embedding
	# take the average of embedding and feed them to activation function
	combined = F.relu((center_h + aggregated) / 4.0)

	return combined


def weight_inter_agg(num_relations, self_feats, neigh_feats, embed_dim, weight, alpha, n, cuda):
	"""
	Weight inter-relation aggregator
	Reference: https://arxiv.org/abs/2002.12307
	:param num_relations: number of relations in the graph
	:param self_feats: batch nodes features or embeddings
	:param neigh_feats: intra-relation aggregated neighbor embeddings for each relation
	:param embed_dim: the dimension of output embedding
	:param weight: parameter used to transform node embeddings before inter-relation aggregation
	:param alpha: weight parameter for each relation used by CARE-Weight
	:param n: number of nodes in a batch
	:param cuda: whether use GPU
	:return: inter-relation aggregated node embeddings
	"""

	# transform batch node embedding and neighbor embedding in each relation with weight parameter
	center_h = torch.mm(self_feats, weight)
	neigh_h = torch.mm(neigh_feats, weight)

	# compute relation weights using softmax
	w = F.softmax(alpha, dim=1)

	# initialize the final neighbor embedding
	if cuda:
		aggregated = torch.zeros(size=(n, embed_dim)).cuda()
	else:
		aggregated = torch.zeros(size=(n, embed_dim))

	# add weighted neighbor embeddings in each relation together
	for r in range(num_relations):
		aggregated += neigh_h[r * n:(r + 1) * n, :] * w[:, r]

	# sum aggregated neighbor embedding and batch node embedding
	# feed them to activation function
	combined = F.relu(center_h + aggregated)

	return combined


def att_inter_agg(num_relations, att_layer, self_feats, neigh_feats, embed_dim, weight, a, n, dropout, training, cuda):
	"""
	Attention-based inter-relation aggregator
	Reference: https://github.com/Diego999/pyGAT
	:param num_relations: num_relations: number of relations in the graph
	:param att_layer: the activation function used by the attention layer
	:param self_feats: batch nodes features or embeddings
	:param neigh_feats: intra-relation aggregated neighbor embeddings for each relation
	:param embed_dim: the dimension of output embedding
	:param weight: parameter used to transform node embeddings before inter-relation aggregation
	:param a: parameters used by attention layer
	:param n: number of nodes in a batch
	:param dropout: dropout for attention layer
	:param training: a flag indicating whether in the training or testing mode
	:param cuda: whether use GPU
	:return combined: inter-relation aggregated node embeddings
	:return att: the attention weights for each relation
	"""

	# transform batch node embedding and neighbor embedding in each relation with weight parameter
	center_h = torch.mm(self_feats, weight)
	neigh_h = torch.mm(neigh_feats, weight)

	import pdb
	pdb.set_trace()
	# compute attention weights
	combined = torch.cat((center_h.repeat(3, 1), neigh_h), dim=1)
	e = att_layer(combined.mm(a))
	attention = torch.cat((e[0:n, :], e[n:2 * n, :], e[2 * n:3 * n, :]), dim=1)
	ori_attention = F.softmax(attention, dim=1)
	attention = F.dropout(ori_attention, dropout, training=training)

	# initialize the final neighbor embedding
	if cuda:
		aggregated = torch.zeros(size=(n, embed_dim)).cuda()
	else:
		aggregated = torch.zeros(size=(n, embed_dim))

	# add neighbor embeddings in each relation together with attention weights
	for r in range(num_relations):
		aggregated += torch.mul(attention[:, r].unsqueeze(1).repeat(1, embed_dim), neigh_h[r * n:(r + 1) * n, :])

	# sum aggregated neighbor embedding and batch node embedding
	# feed them to activation function
	combined = F.relu((center_h + aggregated))

	# extract the attention weights
	att = F.softmax(torch.sum(ori_attention, dim=0), dim=0)

	return combined, att


def threshold_inter_agg(num_relations, self_feats, neigh_feats, embed_dim, weight, threshold, n, cuda):
	"""
	CARE-GNN inter-relation aggregator
	Eq. (9) in the paper
	:param num_relations: number of relations in the graph
	:param self_feats: batch nodes features or embeddings
	:param neigh_feats: intra-relation aggregated neighbor embeddings for each relation
	:param embed_dim: the dimension of output embedding
	:param weight: parameter used to transform node embeddings before inter-relation aggregation
	:param threshold: the neighbor filtering thresholds used as aggregating weights
	:param n: number of nodes in a batch
	:param cuda: whether use GPU
	:return: inter-relation aggregated node embeddings
	"""

	# transform batch node embedding and neighbor embedding in each relation with weight parameter
	center_h = torch.mm(self_feats, weight)
	neigh_h = torch.mm(neigh_feats, weight)

	# initialize the final neighbor embedding
	if cuda:
		aggregated = torch.zeros(size=(n, embed_dim)).cuda()
	else:
		aggregated = torch.zeros(size=(n, embed_dim))

	# add weighted neighbor embeddings in each relation together
	for r in range(num_relations):
		aggregated += neigh_h[r * n:(r + 1) * n, :] * threshold[r]

	# sum aggregated neighbor embedding and batch node embedding
	# feed them to activation function
	combined = F.relu(center_h + aggregated)

	return combined
