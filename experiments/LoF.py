import numpy as np
import argparse
from sklearn.neighbors import LocalOutlierFactor
import scipy.io
from sklearn.metrics import roc_auc_score, average_precision_score
import glob
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='Amazon', help='똑바로 입력해라! [Amazon, YelpChi]')
# parser.add_argument('--n_neighbors', type=float, default=20, help='똑바로 입력해라! [n_neighbors -> int]')
# parser.add_argument('--contamination', type=float, default=0.05, help='똑바로 입력해라! [contamination -> float]')
args = parser.parse_args()

# data_name = args.data
# n_neighbors = args.n_neighbors
# contamination = args.contamination

data_name = args.data
n_neighbors_l = np.arange(5,1000,5)

mat_path = glob.glob(f'./data/{data_name}/{data_name}.mat')[0]
os.makedirs("./results", exist_ok=True)
res = f'./results/{data_name}.txt'

data = scipy.io.loadmat(mat_path)
features = data["features"].toarray()
y_true = data['label'][0]

for n_neighbors in tqdm(n_neighbors_l):
    file = open(res, 'a')
    file.write(f"Num of neighbors: {n_neighbors}\n")
    clf = LocalOutlierFactor(n_neighbors=n_neighbors)

    clf.fit_predict(features)
    X_scores = -clf.negative_outlier_factor_

    auc = roc_auc_score(y_true, X_scores)
    ap = average_precision_score(y_true, X_scores)
    file.write("AUC-ROC: %.5f, AUC-AP: %.5f\n\n" % (auc, ap))
    file.close()