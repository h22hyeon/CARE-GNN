from utils import sparse_to_adjlist
from scipy.io import loadmat
import torch
import pandas as pd

"""
	Read data and save the adjacency matrices to adjacency lists
"""


if __name__ == "__main__":

        # prefix = "./data/pyg/YelpChi"
        # yelp = loadmat(f'{prefix}/raw/YelpChi.mat')
        # net_rur = yelp['net_rur']
        # net_rtr = yelp['net_rtr']
        # net_rsr = yelp['net_rsr']
        # yelp_homo = yelp['homo']

        # sparse_to_adjlist(net_rur, f'{prefix}/processed/yelp_rur_adjlists.pickle')
        # sparse_to_adjlist(net_rtr, f'{prefix}/processed/yelp_rtr_adjlists.pickle')
        # sparse_to_adjlist(net_rsr, f'{prefix}/processed/yelp_rsr_adjlists.pickle')
        # sparse_to_adjlist(yelp_homo, f'{prefix}/processed/yelp_homo_adjlists.pickle')

        prefix = "./data/pyg/AmazonFraud"
        amz = loadmat(f'{prefix}/raw/Amazon.mat')
        net_upu = amz['net_upu']
        net_usu = amz['net_usu']
        net_uvu = amz['net_uvu']
        amz_homo = amz['homo']

        # sparse_to_adjlist(net_upu, f'{prefix}/processed/amazon_upu_adjlists.pickle')
        # sparse_to_adjlist(net_usu, f'{prefix}/processed/amazon_usu_adjlists.pickle')
        # sparse_to_adjlist(net_uvu, f'{prefix}/processed/amazon_uvu_adjlists.pickle')
        # sparse_to_adjlist(amz_homo, f'{prefix}/processed/amazon_homo_adjlists.pickle')
 
        prefix = "./data/pyg/AmazonFraud/processed/"
        data = torch.load(prefix + "AmazonFraud_data.pt")[0]
        data['user'].y[:3305] = 2
        features = data['user'].x.numpy()
        mask_dup = torch.BoolTensor(pd.DataFrame(features).duplicated(keep=False).values)
        data = data.subgraph({'user': ~mask_dup})
        torch.save(data, prefix + 'AmazonFraud_new_data.pt')
        
        sparse_to_adjlist(net_upu[~mask_dup][:, ~mask_dup], prefix + 'amazon_new_upu_adjlists.pickle')
        sparse_to_adjlist(net_usu[~mask_dup][:, ~mask_dup], prefix + 'amazon_new_usu_adjlists.pickle')
        sparse_to_adjlist(net_uvu[~mask_dup][:, ~mask_dup], prefix + 'amazon_new_uvu_adjlists.pickle')
        sparse_to_adjlist(amz_homo[~mask_dup][:, ~mask_dup], prefix + 'amazon_new_homo_adjlists.pickle')