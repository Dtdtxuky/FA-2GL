import torch.utils.data as data
import AtomicProperties
from math import sqrt
import torch
from scipy import stats
import numpy as np
import pandas as pd
import codecs
from subword_nmt.apply_bpe import BPE

TCGA_label_set = ["ALL", "BLCA", "BRCA", "DLBC", "LIHC", "LUAD",
                  "ESCA", "GBM", "HNSC", "KIRC", "LAML", "LCML", "LGG",
                  "LUSC", "MM", "NB", "OV", "PAAD", "SCLC", "SKCM",
                  "STAD", "THCA", 'COAD/READ', 'SARC', 'UCEC', 'MESO', 'PRAD']
# class Data_Encoder(data.Dataset):
#     def __init__(self, list_IDs, labels, df_dti):
#         'Initialization'
#         self.labels = labels
#         self.list_IDs = list_IDs
#         self.df = df_dti

#     def __len__(self):
#         'Denotes the total number of samples'
#         return len(self.list_IDs)

#     def __getitem__(self, index):
#         'Generates one sample of data'
#         # Select sample
#         # Load data and get label
#         index = self.list_IDs[index]
#         d = self.df.iloc[index]['Drug_smile']
#         s = int(self.df.iloc[index]['SE_id'])

#         # d_v = drug2single_vector(d)
#         d_v, input_mask_d = drug2emb_encoder(d)

#         # 副作用的子结构是读取出来的
#         SE_index = np.load("SE_sub_index_50.npy").astype(int)
#         SE_mask = np.load("SE_sub_mask_50.npy")
#         s_v = SE_index[s, :]
#         input_mask_s = SE_mask[s, :]
#         y = self.labels[index]

#         return d_v, s_v, input_mask_d, input_mask_s, y

class Data_Encoder(data.Dataset):
    def __init__(self, list_IDs, IC50, df_dti, PubMed2SMILES, CellLine2Gene, SimilarDrug, SimilarCellLine, PubMedToIndex, CellLineNameToIndex):
        'Initialization'
        self.IC50 = IC50
        self.list_IDs = list_IDs
        self.df = df_dti
        self.PubMed2SMILES = PubMed2SMILES
        self.CellLine2Gene = CellLine2Gene
        self.SimilarDrug = SimilarDrug
        self.SimilarCellLine = SimilarCellLine
        self.PubMedToIndex = PubMedToIndex
        self.CellLineNameToIndex = CellLineNameToIndex

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        '''药物部分'''
        index = self.list_IDs[index]

        # 获取该药物PubMedID
        di = self.df.iloc[index]['PubMedID']

        # 获取SMILES式
        ds = self.PubMed2SMILES[di]

        # 获取特征i向量与Msak矩阵
        d_v, input_mask_d = drug2emb_encoder(ds)


        '''细胞系部分'''
        ci = self.df.iloc[index]['CellLineName']

        # 根据字典获取其基因特征
        GeneFeature = self.CellLine2Gene[ci]
        GeneFeature = torch.tensor(GeneFeature, dtype=torch.float)

        # IC50 信息
        IC50 = self.IC50[index]

        # 药物与细胞系的index信息
        Drug_id = self.PubMedToIndex[di]
        CellLine_id = self.CellLineNameToIndex[ci]

        # 药物与细胞系各自的similar信息
        DrugSim = torch.tensor(self.SimilarDrug[Drug_id])
        CellLineSim = torch.tensor(self.SimilarCellLine[CellLine_id])


        # 类别信息
        DCType = self.df.iloc[index]['Type']
        DCType = TCGA_label_set.index(DCType)

        return d_v, input_mask_d, GeneFeature, IC50, DCType, Drug_id, CellLine_id, DrugSim, CellLineSim


def rmse(y, f):
    rmse = sqrt(((y - f) ** 2).mean())
    return rmse


def pearson(y, f):
    rp = np.corrcoef(y, f)[0, 1]
    return rp


def spearman(y, f):
    rs = stats.spearmanr(y, f)[0]
    return rs

def drug2emb_encoder(smile):
    vocab_path = 'drug_codes_chembl_freq_1500.txt'
    sub_csv = pd.read_csv('subword_units_map_chembl_freq_1500.csv')

    # 初始化一个BPE编码器，该编码器可以用于将文本进行分词或编码
    bpe_codes_drug = codecs.open(vocab_path)
    dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
    idx2word_d = sub_csv['index'].values  # 将所有的子结构列表给提取出来
    words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))  # 构造字典：让子结构与index一一对应

    max_d = 50
    t1 = dbpe.process_line(smile).split()  # split
    try:
        i1 = np.asarray([words2idx_d[i] for i in t1])  # 将该smile的子结构找到对应的index，形成一个index的ndarray
    except:
        i1 = np.array([0])

    l = len(i1)
    if l < max_d:
        i = np.pad(i1, (0, max_d - l), 'constant', constant_values=0)
        input_mask = ([1] * l) + ([0] * (max_d - l))
    else:
        i = i1[:max_d]  # 进行填充
        input_mask = [1] * max_d  # 构造mask（盖住填充部分）

    return i, np.asarray(input_mask)
