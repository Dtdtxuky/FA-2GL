import torch
from DrugFeature import *
from CellLineFeature import *
import torch.nn.functional as F
from BridgeFile import *
from Encoder import Encoder_MultipleLayers, Embeddings


class DrugAndCellLine(torch.nn.Module):
    def __init__(self, device):
        super(DrugAndCellLine, self).__init__()
        self.device = device

        # Drug子结构编码
        input_dim_drug = 2586
        transformer_emb_size_drug = 200
        transformer_dropout_rate = 0.1
        transformer_n_layer_drug = 3
        transformer_intermediate_size_drug = 512
        transformer_num_attention_heads_drug = 8
        transformer_attention_probs_dropout = 0.1
        transformer_hidden_dropout_rate = 0.1

        self.emb = Embeddings(input_dim_drug,
                              transformer_emb_size_drug,
                              50,
                              transformer_dropout_rate)

        self.encoder = Encoder_MultipleLayers(transformer_n_layer_drug,
                                              transformer_emb_size_drug,
                                              transformer_intermediate_size_drug,
                                              transformer_num_attention_heads_drug,
                                              transformer_attention_probs_dropout,
                                              transformer_hidden_dropout_rate)

        self.CrossEncoder = Encoder_MultipleLayers(transformer_n_layer_drug,
                                                   transformer_emb_size_drug,
                                                   transformer_intermediate_size_drug,
                                                   transformer_num_attention_heads_drug,
                                                   transformer_attention_probs_dropout,
                                                   transformer_hidden_dropout_rate)

        # # 混合
        self.FusionCoff = 0.1

        # BridgeGraph
        self.Bridge = BridgeGraph(nodeNum=64, outSize=200, device=self.device, gcnHiddenSizeList=[128, 128, 128],
                                  fcHiddenSizeList=[128])

        self.CellGraph = CellLineGraph(self.device)

    def forward(self, Drug, CellLine):
        DrugMask = Drug[1]
        DrugSim = Drug[2]
        Drug = Drug[0]

        CellSim = CellLine[2]

        # 药物嵌入
        Drug = Drug.long().to(self.device)
        DrugMask = DrugMask.long().to(self.device)
        DrugMask = DrugMask.unsqueeze(1).unsqueeze(2)
        Mask = DrugMask
        DrugMask = (1.0 - DrugMask) * -10000.0
        d0 = self.emb(Drug)

        # 细胞系嵌入
        c0 = self.CellGraph(CellLine[0])

        # 嵌入特征融合
        c1 = c0.view(c0.shape[0], 1, c0.shape[1])
        c1 = c1.repeat(1, 50, 1)
        CrossEncoded_layers = self.CrossEncoder([d0.float(), c1.float()], DrugMask.float(), True)
        d1 = CrossEncoded_layers[0, :, :, :]
        c1 = CrossEncoded_layers[1, :, :, :]
        c1 = self.MaskFeature(c1, Mask)

        # 药物特征Transformer特征
        encoded_layers = self.encoder(d0.float(), DrugMask.float(), False)
        d2 = encoded_layers

        # 特征混合
        d = self.FusionCoff * d1 + (1 - self.FusionCoff) * d2
        d, _ = torch.max(d, dim=1)

        c = self.FusionCoff * c1 + (1 - self.FusionCoff) * c0

        df = d
        cf = c


        d = d.view(d.shape[0], 1, d.shape[1])
        c = c.view(c.shape[0], 1, c.shape[1])


        Score = self.BridgeFusion(d, c, DrugSim, CellSim, df, cf)
        return Score



    def CrossFustionUp(self, d, c):
        # 图前药物编码
        d = self.drugFeature.GetEmbeding(d)

        # 图前细胞系编码
        c = c[0]
        c = c.view(-1, 697, 2).to(self.device)
        c = self.ffn(c).to(self.device)

        # 层归一化
        d = self.norm1(d)
        c = self.norm2(c)

        # 合成一个批次
        Fdc = torch.cat((c, d), dim=1)
        Fdc = self.Att(Fdc)

        # 得到与细胞系特征（与药物原子融合），药物原子特征（与细胞系697个基因融合）
        d = Fdc[:, :697, :]
        c = Fdc[:, 697:, :]

        return d, c

    def BridgeFusion(self, d, c, DrugSim, CellSim, df, cf):
        DrugSim = DrugSim.view(DrugSim.shape[0], 1, DrugSim.shape[1])
        CellSim = CellSim.view(CellSim.shape[0], 1, CellSim.shape[1])

        # d = d.view(d.shape[0], 1, d.shape[1])
        # c = c.view(c.shape[0], 1, c.shape[1])

        Score = self.Bridge(d, c, DrugSim, CellSim, df, cf)

        return Score

    def MaskFeature(self, features, Mask):
        mask = Mask.view(Mask.shape[0], Mask.shape[3], 1)
        mask = mask.repeat(1, 1, features.shape[-1])

        # 将特征张量和掩码张量相乘，将掩盖住的特征设置为零
        masked_features = features * mask.float()  # 将掩码张量转换为 float 类型

        # 计算每个批次的未被掩盖的特征的平均值
        sum_unmasked_features = masked_features.sum(dim=1)  # 沿着第二个维度求和
        sum_unmasked_features = sum_unmasked_features.view(sum_unmasked_features.shape[0], 1,
                                                           sum_unmasked_features.shape[1])
        count_unmasked = mask.sum(dim=1, keepdim=True)  # 统计每个批次未被掩盖的特征数量
        average_unmasked_features = sum_unmasked_features / torch.clamp(count_unmasked, min=1)  # 防止除以零
        average_unmasked_features = average_unmasked_features.view(average_unmasked_features.shape[0], average_unmasked_features.shape[2])

        return average_unmasked_features
