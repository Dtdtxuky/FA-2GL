import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
import pickle
import numpy as np


# class GCN(nn.Module):
#     def __init__(self, inSize, outSize, hiddenSizeList=[], dropout=0.1, bnEveryLayer=False, dpEveryLayer=False,
#                  outBn=False, outAct=False, outDp=False, resnet=True, name='GCN', actFunc=nn.ReLU):
#         super(GCN, self).__init__()
#         self.name = name
#         hiddens, bns = [], []
#         for i, os in enumerate(hiddenSizeList):
#             hiddens.append(nn.Sequential(
#                 nn.Linear(inSize, os),
#             ))
#             bns.append(nn.BatchNorm1d(os))
#             inSize = os
#         bns.append(nn.BatchNorm1d(outSize))
#         self.actFunc = actFunc()
#         self.dropout = nn.Dropout(p=dropout)
#         self.hiddens = nn.ModuleList(hiddens)
#         self.bns = nn.ModuleList(bns)
#         self.out = nn.Linear(inSize, outSize)
#         self.bnEveryLayer = bnEveryLayer
#         self.dpEveryLayer = dpEveryLayer
#         self.outBn = outBn
#         self.outAct = outAct
#         self.outDp = outDp
#         self.resnet = resnet
#
#     def forward(self, x, L):
#         # x: nodeNum × feaSize; L: batchSize × nodeNum × nodeNum
#         for h, bn in zip(self.hiddens, self.bns):
#             a = h(torch.matmul(L, x))  # => batchSize × nodeNum × os
#             if self.bnEveryLayer:
#                 if len(L.shape) == 3:
#                     a = bn(a.transpose(1, 2)).transpose(1, 2)
#                 else:
#                     a = bn(a)
#             a = self.actFunc(a)
#             if self.dpEveryLayer:
#                 a = self.dropout(a)
#             if self.resnet and a.shape == x.shape:
#                 a += x
#             x = a
#         a = self.out(torch.matmul(L, x))  # => batchSize × nodeNum × outSize
#         if self.outBn:
#             if len(L.shape) == 3:
#                 a = self.bns[-1](a.transpose(1, 2)).transpose(1, 2)
#             else:
#                 a = self.bns[-1](a)
#         if self.outAct: a = self.actFunc(a)
#         if self.outDp: a = self.dropout(a)
#         if self.resnet and a.shape == x.shape:
#             a += x
#         x = a
#         return x

class CellLineGraph(nn.Module):
    def __init__(self, device):
        super(CellLineGraph, self).__init__()

        self.device = device
        self.ReLU = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

        self.ffn1 = nn.Linear(697, 512)
        self.norm1 = nn.LayerNorm([512])

        self.ffn2 = nn.Linear(512, 256)
        self.norm2 = nn.LayerNorm([256])

        self.ffn3 = nn.Linear(256, 128)
        self.norm3 = nn.LayerNorm([128])

        self.ffn4 = nn.Linear(128, 100)
        self.norm4 = nn.LayerNorm([100])

        self.ffn5 = nn.Linear(697, 512)
        self.norm5 = nn.LayerNorm([512])

        self.ffn6 = nn.Linear(512, 256)
        self.norm6 = nn.LayerNorm([256])

        self.ffn7 = nn.Linear(256, 128)
        self.norm7 = nn.LayerNorm([128])

        self.ffn8 = nn.Linear(128, 100)
        self.norm8 = nn.LayerNorm([100])



    def forward(self, CellLine):
        
        CellLine = CellLine.to(self.device)

        # CellLineLink = CellLineLink.repeat(len(CellLine), 1, 1).to('cuda')

        # D = torch.eye(CellLineLink.shape[1], dtype=torch.float32, device='cuda').repeat(len(CellLine), 1,1).to('cuda')

        # D[:, range(CellLineLink.shape[1]), range(CellLineLink.shape[1])] = 1 / (torch.sum(CellLineLink, dim=2) ** 0.5).to('cuda')

        # pL = torch.matmul(torch.matmul(D, CellLineLink), D).to('cuda')

        # CellLine = self.norm1(self.ffn(CellLine)).to('cuda')

        # node_gcned = self.nodeGCN(CellLine, pL)

        Exp = CellLine[:,:,0]
        Exp = Exp.view(Exp.shape[0], Exp.shape[1])

        Var = CellLine[:,:,1]
        Var = Var.view(Var.shape[0], Var.shape[1])



        Exp = self.norm1(self.ReLU(self.ffn1(Exp)))
        Exp = self.dropout(Exp)

        Exp = self.norm2(self.ReLU(self.ffn2(Exp)))
        Exp = self.dropout(Exp)

        Exp = self.norm3(self.ReLU(self.ffn3(Exp)))
        Exp = self.dropout(Exp)

        Exp = self.norm4(self.ReLU(self.ffn4(Exp)))


        Var = self.norm5(self.ReLU(self.ffn5(Var)))
        Var = self.dropout(Var)

        Var = self.norm6(self.ReLU(self.ffn6(Var)))
        Var = self.dropout(Var)

        Var = self.norm7(self.ReLU(self.ffn7(Var)))
        Var = self.dropout(Var)

        Var = self.norm8(self.ReLU(self.ffn8(Var)))


        CellLine = torch.cat((Exp,Var),dim=1)


        return CellLine
