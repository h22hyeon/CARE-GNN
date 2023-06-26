import os
import pickle
import numpy as np
import pandas as pd

from typing import Optional
from datetime import datetime
from utils import create_dir

EXP_RES_DIR = "./experimental_results"
DF_VAL_DIR = f"{EXP_RES_DIR}/validation_df"
DF_TEST_DIR = f"{EXP_RES_DIR}/test_df"
LOG_VAL_DIR = f"{EXP_RES_DIR}/validation_log"
LOG_TEST_DIR = f"{EXP_RES_DIR}/test_log"
SAVED_MODEL_DIR = f"{EXP_RES_DIR}/saved_models"
PRED_DIR = f"{EXP_RES_DIR}/predictions"

class ResultManager:
    """
    실험에 사용된 args를 기반으로 실험 결과를 관리하는 역할을 한다.
    실험 결과는 DataFrame과 text 파일 형태로 EXP_RES_DIR 아래에 저장된다.
    학습된 모델 파라미터는 pickle 형태로 SAVED_MODEL_DIR 아래에 저장된다.
    예측 결과는 npy 형태로 PRED_DIR 아래에 저장된다.
    """
    def __init__(self, args) -> None:
        create_dir(SAVED_MODEL_DIR)
        create_dir(PRED_DIR)
        create_dir(EXP_RES_DIR)
        create_dir(DF_VAL_DIR)
        create_dir(DF_TEST_DIR)
        create_dir(LOG_VAL_DIR)
        create_dir(LOG_TEST_DIR)
        
        self.args = args
        data_name = args['data_name']
        model = args['model']
        self.exp_id = f"{model}-{data_name}-{datetime.now().strftime('%y%m%d-%H%M%S-%f')}"
        self.df_val_path = os.path.join(DF_VAL_DIR, f"{self.exp_id}.pkl")
        self.df_test_path = os.path.join(DF_TEST_DIR, f"{model}-{data_name}.pkl")
        self.log_val_path = os.path.join(LOG_VAL_DIR, f"{self.exp_id}.log")
        self.log_test_path = os.path.join(LOG_TEST_DIR, f"{self.exp_id}.log")
        self.model_path = os.path.join(SAVED_MODEL_DIR, f"{self.exp_id}.pickle")
        
        self.df_val = pd.DataFrame()
        self.df_test = pd.read_pickle(self.df_test_path) if os.path.exists(self.df_test_path) else pd.DataFrame()
        self.init_logs()
    
    def load_df_test(self):
        df_test = pd.DataFrame()
        pair_name = f"{self.args['model']}-{self.args['data_name']}"
        log_test_path_l = [os.path.join(LOG_TEST_DIR, filename) for filename in os.listdir(LOG_TEST_DIR) if pair_name in filename]
        for path in log_test_path_l:
            with open(path, 'r') as f:
                idx = len(df_test)
                lines = [line.strip() for line in f.readlines()][:-1]
                result = lines.pop()
                if (not 'Test performance' in result):
                    continue
                exp_id = path.split('/')[-1][:-4]
                df_test.loc[idx, 'exp_id'] = exp_id
                result = dict([tuple(metric.strip().split(': ')) for metric in result.split('- ')[1:]])
                result = dict([(k.lower(), float(v)) for (k, v) in result.items()])
                df_test.loc[idx, 'epoch_best'] = result['epoch_best']
                df_test.loc[idx, 'accuracy'] = result['accuracy']
                df_test.loc[idx, 'f1'] = result['f1']
                df_test.loc[idx, 'f1_macro'] = result['f1-macro']
                df_test.loc[idx, 'precision'] = result['precision']
                df_test.loc[idx, 'precision_macro'] = result['ap']
                df_test.loc[idx, 'recall'] = result['recall']
                df_test.loc[idx, 'recall_macro'] = result['recall-macro']
                df_test.loc[idx, 'auc'] = result['auc-roc']
                args = dict([tuple(line.split(': ')) for line in lines])
                for key in sorted(args.keys()):
                    df_test.loc[idx, key] = args[key]
        df_test.to_pickle(self.df_test_path)
        self.df_test = df_test
    
    def get_configuration_line(self) -> str:
        line = ""
        for key in sorted(self.args.keys()):
            line = f"{line}\n{key}: {self.args[key]}"
        return line
    
    def init_logs(self):
        line = self.get_configuration_line()[1:]
        with open(self.log_val_path, 'a') as file:
            file.write(line + "\n")
        with open(self.log_test_path, 'a') as file:
            file.write(line + "\n")
    
    def write_val_log(self, epoch: int, epoch_best: int, accuracy: float,
                      f1: float, f1_macro: float, 
                      precision: float, precision_macro: float,
                      recall: float, recall_macro: float, auc: float, line: str,
                      print_line: bool=True) -> None:
        with open(self.log_val_path, 'a') as file:
            line = f"[Epoch-{str(epoch).zfill(3)}] Validation performance\n{line}"
            file.write(line + "\n")
            if print_line: print(line)
        
        idx = len(self.df_val)
        self.df_val.loc[idx, 'epoch'] = epoch
        self.df_val.loc[idx, 'epoch_best'] = epoch_best
        self.df_val.loc[idx, 'accuracy'] = accuracy
        self.df_val.loc[idx, 'f1'] = f1
        self.df_val.loc[idx, 'f1_macro'] = f1_macro
        self.df_val.loc[idx, 'precision'] = precision
        self.df_val.loc[idx, 'precision_macro'] = precision_macro
        self.df_val.loc[idx, 'recall'] = recall
        self.df_val.loc[idx, 'recall_macro'] = recall_macro
        self.df_val.loc[idx, 'auc'] = auc
        self.df_val.to_pickle(self.df_val_path)
            
    def write_test_log(self, epoch_best: int, accuracy: float,
                       f1: float, f1_macro: float,
                       precision: float, precision_macro: float,
                       recall: float, recall_macro: float, auc: float, line: str,
                       print_line: bool=True) -> None:
        self.load_df_test()
        with open(self.log_test_path, 'a') as file:
            line = f"Test performance: - Epoch_Best: {epoch_best}\t"+ line
            file.write(line + "\n")
            if print_line: print(line)
            
        idx = len(self.df_test)
        self.df_test.loc[idx, 'exp_id'] = self.exp_id
        self.df_test.loc[idx, 'epoch_best'] = epoch_best
        self.df_test.loc[idx, 'accuracy'] = accuracy
        self.df_test.loc[idx, 'f1'] = f1
        self.df_test.loc[idx, 'f1_macro'] = f1_macro
        self.df_test.loc[idx, 'precision'] = precision
        self.df_test.loc[idx, 'precision_macro'] = precision_macro
        self.df_test.loc[idx, 'recall'] = recall
        self.df_test.loc[idx, 'recall_macro'] = recall_macro
        self.df_test.loc[idx, 'auc'] = auc
        for key in sorted(self.args.keys()):
            self.df_test.loc[idx, key] = self.args[key]
        self.df_test.to_pickle(self.df_test_path)
    
    def get_best_model_exp_id(self, metric: Optional[str]='auc') -> str:
        '''
        주어진 model, dataset pair에 대해서 metric 기준으로 가장 performance가 좋았던 exp_id를 반환한다.
        
        [Possible metric]: 'accuracy', 'f1', 'f1_macro', 'precision', 'precision_macro', 'recall', 'recall_macro', 'auc' (default)
        '''
        return self.df_test.iloc[self.df_test[metric].argmax()]['exp_id']
    
    def get_best_model_path(self, metric: Optional[str]='auc') -> str:
        '''
        주어진 model, dataset pair에 대해서 metric 기준으로 가장 performance가 좋았던 model의 경로를 반환한다.
        
        [Possible metric]: 'accuracy', 'f1', 'f1_macro', 'precision', 'precision_macro', 'recall', 'recall_macro', 'auc' (default)
        '''
        return os.path.join(SAVED_MODEL_DIR, f'{self.get_best_model_exp_id(metric)}.pickle')
    
    def save_predictions(self, arr: np.ndarray, name: str) -> None:
        np.save(os.path.join(PRED_DIR, f'{self.exp_id}-{name}'), arr)