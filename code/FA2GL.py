import argparse
import random, os, sys
import csv
from Net import *
from sklearn.model_selection import KFold
import pandas as pd
import hickle as hkl
import pubchempy as pcp
import pickle
import torch.utils.data as data
from utils import *
import time

unit_list = [256]
israndom = False
Drug_info_file = '../data/drug/1.Drug_listMon Jun 24 09_00_55 2019.csv'
Cell_line_info_file = '../data/CCLE/Cell_lines_annotations_20181226.txt'
Drug_feature_file = '../data/drug/drug_graph_feat'
Cancer_response_exp_file = '../data/CCLE/GDSC_IC50.csv'
PPI_file = "../data/PPI/PPI_network.txt"
selected_info_common_cell_lines = "../data/CCLE/cellline_list.txt"
selected_info_common_genes = "../data/CCLE/gene_list.txt"
celline_feature_folder = "../data/CCLE/omics_data"

Max_atoms = 100
TCGA_label_set = ["ALL", "BLCA", "BRCA", "DLBC", "LIHC", "LUAD",
                  "ESCA", "GBM", "HNSC", "KIRC", "LAML", "LCML", "LGG",
                  "LUSC", "MM", "NB", "OV", "PAAD", "SCLC", "SKCM",
                  "STAD", "THCA", 'COAD/READ', 'SARC', 'UCEC', 'MESO', 'PRAD']


def loss_fun(label, pred, device):
    label = label.to(device)
    pred = pred.to(device)
    loss = torch.sum((label - pred) ** 2)
    return loss


def MetadataGenerate(Drug_info_file, Cell_line_info_file, Drug_feature_file, PPI_file, selected_info_common_cell_lines,
                     selected_info_common_genes):
    with open(selected_info_common_cell_lines) as f:
        common_cell_lines = [item.strip() for item in f.readlines()]

    with open(selected_info_common_genes) as f:
        common_genes = [item.strip() for item in f.readlines()]
    idx_dic = {}
    for index, item in enumerate(common_genes):
        idx_dic[item] = index

    ppi_adj_info = [[] for item in common_genes]
    for line in open(PPI_file).readlines():
        gene1, gene2 = line.split('\t')[0], line.split('\t')[1]
        if idx_dic[gene1] <= idx_dic[gene2]:
            ppi_adj_info[idx_dic[gene1]].append(idx_dic[gene2])
            ppi_adj_info[idx_dic[gene2]].append(idx_dic[gene1])

    reader = csv.reader(open(Drug_info_file, 'r'))
    rows = [item for item in reader]
    drugid2pubchemid = {item[0]: item[5] for item in rows if item[5].isdigit()}

    cellline2cancertype = {}
    for line in open(Cell_line_info_file).readlines()[1:]:
        cellline_id = line.split('\t')[1]
        TCGA_label = line.strip().split('\t')[-1]
        cellline2cancertype[cellline_id] = TCGA_label

    drug_pubchem_id_set = []
    drug_feature = {}
    for each in os.listdir(Drug_feature_file):
        drug_pubchem_id_set.append(each.split('.')[0])
        feat_mat, adj_list, degree_list = hkl.load('%s/%s' % (Drug_feature_file, each))
        drug_feature[each.split('.')[0]] = [feat_mat, adj_list, degree_list]
    assert len(drug_pubchem_id_set) == len(drug_feature.values())

    IC50_df = pd.read_csv(Cancer_response_exp_file, sep=',', header=0, index_col=[0])
    drug_match_list = [item for item in IC50_df.index if item.split(':')[1] in drugid2pubchemid.keys()]
    IC50_df = IC50_df.loc[drug_match_list]

    index_name = [drugid2pubchemid[item.split(':')[1]] for item in IC50_df.index if
                  item.split(':')[1] in drugid2pubchemid.keys()]
    IC50_df.index = index_name
    redundant_names = list(set([item for item in IC50_df.index if list(IC50_df.index).count(item) > 1]))
    retain_idx = []
    for i in range(len(IC50_df.index)):
        if IC50_df.index[i] not in redundant_names:
            retain_idx.append(i)
    IC50_df = IC50_df.iloc[retain_idx]

    data_idx = []
    for each_drug in IC50_df.index:
        for each_cellline in IC50_df.columns:
            if str(each_drug) in drug_pubchem_id_set and each_cellline in common_cell_lines:
                if not np.isnan(
                        IC50_df.loc[each_drug, each_cellline]) and each_cellline in cellline2cancertype.keys() and \
                        cellline2cancertype[each_cellline] in TCGA_label_set:
                    ln_IC50 = float(IC50_df.loc[each_drug, each_cellline])
                    data_idx.append((each_cellline, each_drug, ln_IC50, cellline2cancertype[each_cellline]))
    nb_celllines = len(set([item[0] for item in data_idx]))
    nb_drugs = len(set([item[1] for item in data_idx]))
    print('%d instances across %d cell lines and %d drugs were generated.' % (len(data_idx), nb_celllines, nb_drugs))
    return ppi_adj_info, data_idx, drug_pubchem_id_set


def DataSplit(data_idx, TCGA_label_set, n_splits):
    # n_split: number of CV
    # data_train_idx,data_test_idx = [[] for i in range(n_splits)] , [[] for i in range(n_splits)]
    data_train_idx = []
    data_test_idx = []
    kind = 0
    for each_type in TCGA_label_set:
        data_subtype_idx = [item for item in data_idx if item[-1] == each_type]
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=123)
        idx = 0
        for train, test in kf.split(data_subtype_idx):
            if kind == 0:
                data_train_idx.append([data_subtype_idx[item] for item in train])
                data_test_idx.append([data_subtype_idx[item] for item in test])
            else:
                data_train_idx[idx] = data_train_idx[idx] + [data_subtype_idx[item] for item in train]
                data_test_idx[idx] = data_test_idx[idx] + [data_subtype_idx[item] for item in test]
            idx += 1
        kind = kind + 1
    return data_train_idx, data_test_idx


def FeatureExtract(data_idx, celline_feature_folder, selected_info_common_cell_lines, selected_info_common_genes):
    cancer_type_list = []
    nb_instance = len(data_idx)
    drug_data = [[] for item in range(nb_instance)]
    cell_line_data_feature = [[] for item in range(nb_instance)]
    target = np.zeros(nb_instance, dtype='float32')
    cellline_drug_pair = []
    with open(selected_info_common_cell_lines) as f:
        common_cell_lines = [item.strip() for item in f.readlines()]

    with open(selected_info_common_genes) as f:
        common_genes = [item.strip() for item in f.readlines()]
    dic_cell_line_feat = {}
    for each in common_cell_lines:
        dic_cell_line_feat[each] = pd.read_csv('%s/%s.csv' % (celline_feature_folder, each), index_col=0).loc[
            common_genes].values

    return dic_cell_line_feat


def FunPubMed2SMILES(PubMedIDSet):
    # PubMed2SMILES = FunPubMed2SMILES(drug_pubchem_id_set)

    # with open('PubMed2SMILES.pkl', 'wb') as f:
    #     pickle.dump(PubMed2SMILES, f)

    dicPubMed2SMILES = {}
    i = 1
    for PubMedID in PubMedIDSet:
        compound = pcp.Compound.from_cid(PubMedID)
        SMILES = compound.canonical_smiles
        dicPubMed2SMILES[PubMedID] = SMILES
        i = i + 1

    return dicPubMed2SMILES


def ToCellLineGraph(GeneF, GeneE):
    # 构造所用的CellLine的图数据
    batch = GeneF.shape[0]
    GeneF = GeneF.view(-1, 2)

    # 定义增量列表
    increments = [697 * i for i in range(0, batch)]

    # 迭代每个批次并添加相应的增量
    for batch_index, increment in enumerate(increments):
        # 选择当前批次的数据
        batch_tensor = GeneE[batch_index]
        # 添加增量
        batch_tensor += increment

    GeneE = GeneE.view(2, -1)

    return GeneF, GeneE


def Trainfun(model, device, train_loader, optimizer, epoch, test_loader):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    avg_loss = []
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()

    for batch_idx, Mydata in enumerate(train_loader):
        optimizer.zero_grad()

        batch = Mydata[0].shape[0]

        DrugIndex = Mydata[5]
        CellLineIndex = Mydata[6]
        DrugSim = Mydata[7]
        CellLineSim = Mydata[8]

        # 获取基本数据
        DrugIndex = DrugIndex.unsqueeze(0).expand(batch, -1)
        DrugSim = torch.gather(DrugSim, 1, DrugIndex)
        CellLineIndex = CellLineIndex.unsqueeze(0).expand(batch, -1)
        CellLineSim = torch.gather(CellLineSim, 1, CellLineIndex)

        # 获取基本数据
        Drug = (Mydata[0], Mydata[1], DrugSim)
        CellLine = (Mydata[2], Mydata[2].shape[0], DrugSim)

        IC50 = Mydata[3]

        # 模型运行
        out = model(Drug, CellLine)

        # 损失函数
        pred = out.to(device)

        total_preds = torch.cat((total_preds, pred.cpu()), 0)
        total_labels = torch.cat((total_labels, IC50.cpu()), 0)

        loss = loss_fun(IC50, pred.flatten(), device).to('cpu')
        loss.backward()
        optimizer.step()

        avg_loss.append(loss.item())

        if batch_idx % 100 == 0:
            print('Epoch:', epoch, 'Loss', sum(avg_loss) / len(avg_loss))

    return sum(avg_loss) / len(
        avg_loss), total_labels.detach().numpy().flatten(), total_preds.detach().numpy().flatten()


def Predict(model, device, train_loader, optimizer, epoch, test_loader):
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()

    model.eval()
    print('Testing on {} samples...'.format(len(test_loader.dataset)))

    for batch_idx, Mydata in enumerate(test_loader):
        batch = Mydata[0].shape[0]

        DrugIndex = Mydata[5]
        CellLineIndex = Mydata[6]
        DrugSim = Mydata[7]
        CellLineSim = Mydata[8]

        # 获取基本数据
        DrugIndex = DrugIndex.unsqueeze(0).expand(batch, -1)
        DrugSim = torch.gather(DrugSim, 1, DrugIndex)
        CellLineIndex = CellLineIndex.unsqueeze(0).expand(batch, -1)
        CellLineSim = torch.gather(CellLineSim, 1, CellLineIndex)

        # 获取基本数据
        Drug = (Mydata[0], Mydata[1], DrugSim)
        CellLine = (Mydata[2], Mydata[2].shape[0], DrugSim)

        IC50 = Mydata[3]

        # 模型运行
        out = model(Drug, CellLine)

        # 损失函数
        pred = out.to(device)

        total_preds = torch.cat((total_preds, pred.cpu()), 0)
        total_labels = torch.cat((total_labels, IC50.cpu()), 0)


        if batch_idx % 100 == 0:
            print(batch_idx)

    return total_labels.detach().numpy().flatten(), total_preds.detach().numpy().flatten()


def Evalution(label, pred):
    Myrmse = rmse(label, pred)
    Mypearson = pearson(label, pred)
    Myspearman = spearman(label, pred)

    print(Myrmse, Mypearson, Myspearman)

    return Myrmse, Mypearson, Myspearman


def main(training_generator, testing_generator, modeling, lr, num_epoch, weight_decay):
    print('Learning rate: ', lr)
    print('Epochs: ', num_epoch)

    print('CPU/GPU: ', torch.cuda.is_available())
    device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
    print('Device: ', device)
    model = modeling(device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(num_epoch):
        train_loss, label, pred = Trainfun(model=model, device=device,
                                           train_loader=training_generator,
                                           optimizer=optimizer,
                                           epoch=epoch + 1, test_loader=testing_generator)
        x, y, z = Evalution(label, pred)

        if epoch % 5 == 0:
            checkpointsFolder = 'checkpoints/'
            torch.save(model.state_dict(), checkpointsFolder + '1:' + str(epoch))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('--train_batch', type=int, required=False, default=32, help='Batch size training set')
    parser.add_argument('--lr', type=float, required=False, default=1e-4, help='Learning rate')
    parser.add_argument('--wd', type=float, required=False, default=0.01, help='weight_decay')
    parser.add_argument('--epoch', type=int, required=False, default=300, help='Number of epoch')
    parser.add_argument('--log_interval', type=int, required=False, default=40, help='Log interval')
    parser.add_argument('--cuda_name', type=str, required=False, default='cuda', help='Cuda')
    parser.add_argument('--save_model', action='store_true', default=True, help='save model and features')
    args = parser.parse_args()

    lr = args.lr
    num_epoch = args.epoch
    weight_decay = args.wd
    log_interval = args.log_interval
    cuda_name = args.cuda_name
    save_model = args.save_model

    # 模型选择
    modeling = DrugAndCellLine

    np.random.seed(123)
    random.seed(123)

    # data_idx:{细胞系，药物PubMed编号，IC50，癌症类型}
    ppi_adj_info, data_idx, drug_pubchem_id_set = MetadataGenerate(Drug_info_file, Cell_line_info_file,
                                                                   Drug_feature_file, PPI_file,
                                                                   selected_info_common_cell_lines,
                                                                   selected_info_common_genes)


    with open('CellLine2Gene_gyh.pkl', 'rb') as f:
        CellLine2Gene = pickle.load(f)

    # PubMedID对应的SMILES信息
    with open('PubMed2SMILES.pkl', 'rb') as f:
        PubMed2SMILES = pickle.load(f)

    # with open('Distance.pickle', 'rb') as f:
    #     Distance = pickle.load(f)

    with open('PubMedToIndex.pkl', 'rb') as f:
        PubMedToIndex = pickle.load(f)

    with open('CellLineNameToIndex.pkl', 'rb') as f:
        CellLineNameToIndex = pickle.load(f)

    with open('SimilarDrug.pkl', 'rb') as f:
        SimilarDrug = pickle.load(f)

    with open('SimilarCellLine.pkl', 'rb') as f:
        SimilarCellLine = pickle.load(f)

    # 划分数据集
    data_train_idx, data_test_idx = DataSplit(data_idx, TCGA_label_set, n_splits=5)

    params = {'batch_size': 32,
              'shuffle': True}

    params1 = {'batch_size': 32,
               'shuffle': True}

    for k in range(5):
        data_train = data_train_idx[k]
        data_test = data_test_idx[k]

        data_train = [list(train) for train in data_train]
        data_test = [list(test) for test in data_test]

        df_train = pd.DataFrame(data=data_train, columns=['CellLineName', 'PubMedID', 'IC50', 'Type'])
        df_test = pd.DataFrame(data=data_test, columns=['CellLineName', 'PubMedID', 'IC50', 'Type'])

        # 创建数据集和数据加载器
        training_set = Data_Encoder(df_train.index.values, df_train.IC50.values, df_train, PubMed2SMILES, CellLine2Gene,
                                    SimilarDrug, SimilarCellLine, PubMedToIndex, CellLineNameToIndex)
        testing_set = Data_Encoder(df_test.index.values, df_test.IC50.values, df_test, PubMed2SMILES, CellLine2Gene,
                                   SimilarDrug, SimilarCellLine, PubMedToIndex, CellLineNameToIndex)

        training_generator = torch.utils.data.DataLoader(training_set, **params)
        testing_generator = torch.utils.data.DataLoader(testing_set, **params1)

        main(training_generator, testing_generator, modeling, lr, num_epoch, weight_decay)
