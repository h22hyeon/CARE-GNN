import torch
import torch.nn as nn
from torch.nn import init


"""
	CARE-GNN Models
	Paper: Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters
	Source: https://github.com/YingtongDou/CARE-GNN
"""


class OneLayerCARE(nn.Module):
	"""
	The CARE-GNN model in one layer
	"""

	def __init__(self, num_classes, inter1, lambda_1):
		"""
		Initialize the CARE-GNN model
		:param num_classes: number of classes (2 in our paper)
		:param inter1: the inter-relation aggregator that output the final embedding
		"""
		super(OneLayerCARE, self).__init__()
		self.inter1 = inter1
		self.xent = nn.CrossEntropyLoss()

		# the parameter to transform the final embedding
		self.weight = nn.Parameter(torch.FloatTensor(inter1.embed_dim, num_classes))
		init.xavier_uniform_(self.weight)
		self.lambda_1 = lambda_1

	def forward(self, nodes, labels, train_flag=True):
		# InterAgg 레이어는 최종적으로 노드의 inter-agrregated embedding과 label-aware score를 반환한다.
		embeds1, label_scores = self.inter1(nodes, labels, train_flag)
		scores = torch.mm(embeds1, self.weight) # 이를 최종적으로 inter-agrregated embedding을 GNN score로 변환한다.
		return scores, label_scores

	def to_prob(self, nodes, labels, train_flag=True):
		# OneLayerCARE 레이어는 최종적으로 노드의 GNN score와 label-aware score를 반환한다.
		gnn_scores, label_scores = self.forward(nodes, labels, train_flag)
		# 두 score를 softmax 함수를 통해 확률로 변환한다.
		gnn_prob = nn.functional.softmax(gnn_scores, dim=1)
		label_prob = nn.functional.softmax(label_scores, dim=1)
		return gnn_prob, label_prob

	def loss(self, nodes, labels, train_flag=True):
		gnn_scores, label_scores = self.forward(nodes, labels, train_flag)
		# Simi loss, Eq. (4) in the paper
		label_loss = self.xent(label_scores, labels.squeeze()) # positive sample의 label aware score를 이용한 loss (latent space로 투영하는 가중치 행렬도 학습됨.)
		# GNN loss, Eq. (10) in the paper
		gnn_loss = self.xent(gnn_scores, labels.squeeze()) # GNN score를 이용한 loss
		# the loss function of CARE-GNN, Eq. (11) in the paper
		final_loss = gnn_loss + self.lambda_1 * label_loss
		return final_loss
