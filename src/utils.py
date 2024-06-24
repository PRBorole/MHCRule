import numpy as np
import pandas as pd
import re
import random
from tqdm import tqdm
import time
from Bio import SeqIO
import pickle
import itertools

from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, roc_curve, auc
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import train_test_split
import aaanalysis as aa
aa.options["verbose"] = False
aa.options["random_state"] = 42

class aa_(): 
    aa_str = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ-'
    aa_dict = {i:idx for idx,i in enumerate(aa_str)}
    
    def __setattr__(self, name, value):
        raise AttributeError("Constants are immutable")
        
        
def set_seed(seed):
    """Set seed for reproducibility.

    Args:
    seed (int): Seed value.
    """
    # Set seed for random module
    random.seed(seed)
    
    # Set seed for numpy
    np.random.seed(seed)

def fasta_to_dataframe(fasta_file):
    """
    Converts a FASTA file into a pandas DataFrame

    Parameters:
    - fasta_file (str): Path to the FASTA file to be converted

    Returns:
    - pandas.DataFrame: DataFrame containing sequence IDs, sequences, HLA names and HLA lengths
    """
    records = SeqIO.parse(fasta_file, "fasta")
    data = [(record.id, str(record.seq), 
             record.description.split()[1], 
             record.description.split()[2]) for record in records]
    df = pd.DataFrame(data, columns=['ID', 'Sequence','HLA', 'length'])
    return df

def auprc_scorer(y_true, y_pred_proba):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    return auc(recall, precision)


def get_vector_representation(sequence_ls):
    '''
    Args:
        sequence_ls (list): sequence list to convert
    Returns:
        (np.array): List of sequence string converted to vectors 
    '''
    if type(sequence_ls)==pd.Series:
        sequence_ls = sequence_ls.to_list()
    return np.array([[aa for aa in sequence] for sequence in sequence_ls])


def get_sequence_encoding(sequence_ls, scale=None):
    '''
    Args:
        sequence_ls (np.array): array of sequence vectors
    Returns:
        (np.array): encoded peptide array
    '''
    # if sequences are not vectorized yet,
    if type(sequence_ls)!=np.ndarray:
        sequence_ls = self.get_vector_representation(sequence_ls)

    # get scale 
    df_scales = aa.load_scales()
    if scale==None:
        print("error: scale not selected")
    scale_series = df_scales[scale]

    return np.array([[scale_series.get(x, 0) for x in row] for row in sequence_ls])


def get_metric_results(y_true=None, y_predicted=None, y_predicted_proba=None, metrics=['AUROC', 'AUPRC']):
    '''
    Args:
        y_true (np.array or list): array or list of true labels
        y_predicted (np.array or list): array or list of predicted labels
        y_predicted_proba (np.array or list): array or list of predicted probability
        metrics (list): List of metrics to be calculated
    Returns:
        metrics_results (list): list of calculated metrics result
    '''
    metrics_results = []
    
    for metric in metrics:
        if 'accuracy' == metric:
            accuracy_ = accuracy_score(y_true, y_predicted)
            metrics_results = metrics_results + [accuracy_]

        if 'f1' == metric:
            f1_ = f1_score(y_true, y_predicted)
            metrics_results = metrics_results + [f1_]

        if 'AUPRC' == metric:
            precision, recall, _ = precision_recall_curve(y_true, y_predicted_proba)
            auprc_score = auc(recall, precision)
            metrics_results = metrics_results + [auprc_score]

        if 'AUROC' == metric:
            fpr, tpr, _ = roc_curve(y_true, y_predicted_proba)
            auroc_score = auc(fpr, tpr)
            metrics_results = metrics_results + [auroc_score]
        
    return metrics_results


