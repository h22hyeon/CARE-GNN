import time
import os
import random
import argparse
from sklearn.model_selection import train_test_split

from utils import *
from model import *
from layers import *
from graphsage import *
from result_manager import *

"""
        Training CARE-GNN
        Paper: Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters
        Source: https://github.com/YingtongDou/CARE-GNN
"""

class ModelHandler(object):
        def __init__(self, config):
                self.result = ResultManager(args=config) # TODO
                self.args = argparse.Namespace(**config)
  
        def train(self):
                args = self.args
                # args.cuda = not args.no_cuda and torch.cuda.is_available()
                # device = torch.device(args.cuda_id)
                # torch.cuda.set_device(device)

                # load graph, feature, and label
                homo, relation_list, feat_data, labels = load_data(args.data_name) # KDK 데이터 셋에서 realation의 수가 달라질 수 있어 수정함.

                # train/validation/test set 분할.
                np.random.seed(args.seed)
                random.seed(args.seed)
                # Label의 비율을 동일하게 가져가기 위해 계층적 샘플링을 수행한다.
                if args.data_name.startswith('amazon'):
                        idx_unlabeled = 2013 if args.data_name == 'amazon_new' else 3305
                        index = list(range(idx_unlabeled, len(labels)))
                        idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels[idx_unlabeled:], stratify=labels[idx_unlabeled:], train_size=args.train_ratio, random_state=args.seed, shuffle=True)
                        idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest, test_size=args.test_ratio, random_state=args.seed, shuffle=True)
                        print(idx_train[:10])

                elif args.data_name == 'yelp':
                        index = list(range(len(labels)))
                        idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels, stratify=labels, train_size=args.train_ratio, random_state=args.seed, shuffle=True)
                        idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest, test_size=args.test_ratio, random_state=args.seed, shuffle=True)
                        print(args.seed)

                print(f'Run on {args.data_name}, postive/total num: {np.sum(labels)}/{len(labels)}, train num {len(y_train)},'+
                                        f'valid num {len(y_valid)}, test num {len(y_test)}, test positive num {np.sum(y_test)}')

                # split pos neg sets for under-sampling
                train_pos, train_neg = pos_neg_split(idx_train, y_train) # Training set에서 positive/negative sample에 대한 인덱스를 구분한다.

                # initialize model input
                features = nn.Embedding(feat_data.shape[0], feat_data.shape[1])
                if args.data_name.startswith('amazon'):
                        feat_data = normalize(feat_data)
                features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False) # feature를 임베딩한다 (고정).
                features = features.cuda()


                adj_lists = relation_list

                print(f'Model: {args.model}, emb_size: {args.emb_size}.')

                # build one-layer models
                if args.model == 'CARE':
                        if (args.data_name == "yelp") or (args.data_name.startswith('amazon')):
                                intra1 = IntraAgg(features, feat_data.shape[1], cuda=True)
                                intra2 = IntraAgg(features, feat_data.shape[1], cuda=True)
                                intra3 = IntraAgg(features, feat_data.shape[1], cuda=True)
                                inter1 = InterAgg3(features, feat_data.shape[1], args.emb_size, adj_lists, [intra1, intra2, intra3], step_size=args.step_size, cuda=True)
                        else:
                                intra1 = IntraAgg(features, feat_data.shape[1], cuda=args.cuda)
                                inter1 = InterAgg1(features, feat_data.shape[1], args.emb_size, adj_lists, [intra1], step_size=args.step_size, cuda=True)
                if args.model == 'CARE':
                        model = OneLayerCARE(2, inter1, args.lambda_1)
                model = model.cuda()

                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
                times = []

                auc_best, f1_mac_best, epoch_best = 1e-10, 1e-10, 0

                # train the model
                for epoch in range(args.epochs):
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
                        for batch in range(num_batches):
                                start_time = time.time()
                                i_start = batch * args.batch_size
                                i_end = min((batch + 1) * args.batch_size, len(sampled_idx_train))
                                batch_nodes = sampled_idx_train[i_start:i_end]
                                batch_label = labels[np.array(batch_nodes)]
                                optimizer.zero_grad()

                                loss = model.loss(batch_nodes, Variable(torch.cuda.LongTensor(batch_label)))
                                
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
                                print("Valid at epoch {}".format(epoch))
                                auc_val, recall_val, f1_mac_val, precision_val = test(idx_valid, y_valid, model, self.args.batch_size, self.result, epoch, epoch_best, flag="val")
                                gain_auc = (auc_val - auc_best)/auc_best
                                gain_f1_mac =  (f1_mac_val - f1_mac_best)/f1_mac_best
                                if (gain_auc + gain_f1_mac) > 0:
                                        gnn_recall_best, f1_mac_best, auc_best, epoch_best = recall_val, f1_mac_val, auc_val, epoch
                                        torch.save(model.state_dict(), self.result.model_path)
                        if (epoch - epoch_best) > self.args.patience:
                                line = f"Early stopping at epoch {epoch}"
                                print(line)
                                break

                print("Restore model from epoch {}".format(epoch_best))
                model.load_state_dict(torch.load(self.result.model_path))
                auc_test, recall_test, f1_mac_test, precision_test = test(idx_test, y_test, model, self.args.batch_size, self.result, epoch_best=epoch_best, flag="test")
                return auc_test, recall_test, f1_mac_test