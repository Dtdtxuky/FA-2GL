import torch.nn as nn
import torch
import numpy as np


class TextEmbedding(nn.Module):
    def __init__(self, embedding, dropout=0.3, freeze=True, name='textEmbedding'):
        super(TextEmbedding, self).__init__()
        self.name = name
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding, dtype=torch.float32), freeze=freeze)
        self.dropout1 = nn.Dropout(p=dropout / 2)
        self.dropout2 = nn.Dropout(p=dropout / 2)
        self.p = dropout

    def forward(self, x):
        # x: batchSize × seqLen
        if self.p > 0:
            x = self.dropout2(self.dropout1(self.embedding(x)))
        else:
            x = self.embedding(x)
        return x


class GCN(nn.Module):
    def __init__(self, inSize, outSize, hiddenSizeList=[], dropout=0.1, bnEveryLayer=False, dpEveryLayer=False,
                 outBn=False, outAct=False, outDp=False, resnet=True, name='GCN', actFunc=nn.ReLU):
        super(GCN, self).__init__()
        self.name = name
        hiddens, bns = [], []
        for i, os in enumerate(hiddenSizeList):
            hiddens.append(nn.Sequential(
                nn.Linear(inSize, os),
            ))
            bns.append(nn.BatchNorm1d(os))
            inSize = os
        bns.append(nn.BatchNorm1d(outSize))
        self.actFunc = actFunc()
        self.dropout = nn.Dropout(p=dropout)
        self.hiddens = nn.ModuleList(hiddens)
        self.bns = nn.ModuleList(bns)
        self.out = nn.Linear(inSize, outSize)
        self.bnEveryLayer = bnEveryLayer
        self.dpEveryLayer = dpEveryLayer
        self.outBn = outBn
        self.outAct = outAct
        self.outDp = outDp
        self.resnet = resnet

    def forward(self, x, L):
        # x: nodeNum × feaSize; L: batchSize × nodeNum × nodeNum
        for h, bn in zip(self.hiddens, self.bns):
            a = h(torch.matmul(L, x))  # => batchSize × nodeNum × os
            if self.bnEveryLayer:
                if len(L.shape) == 3:
                    a = bn(a.transpose(1, 2)).transpose(1, 2)
                else:
                    a = bn(a)
            a = self.actFunc(a)
            if self.dpEveryLayer:
                a = self.dropout(a)
            if self.resnet and a.shape == x.shape:
                a += x
            x = a
        a = self.out(torch.matmul(L, x))  # => batchSize × nodeNum × outSize
        if self.outBn:
            if len(L.shape) == 3:
                a = self.bns[-1](a.transpose(1, 2)).transpose(1, 2)
            else:
                a = self.bns[-1](a)
        if self.outAct: a = self.actFunc(a)
        if self.outDp: a = self.dropout(a)
        if self.resnet and a.shape == x.shape:
            a += x
        x = a
        return x


class MLP(nn.Module):
    def __init__(self, inSize, outSize, hiddenList=[], dropout=0.0, bnEveryLayer=False, dpEveryLayer=False, outBn=False,
                 outAct=False, outDp=False, name='MLP', actFunc=nn.ReLU):
        super(MLP, self).__init__()
        self.name = name
        hiddens, bns = [], []
        for i, os in enumerate(hiddenList):
            hiddens.append(nn.Sequential(
                nn.Linear(inSize, os),
            ))
            bns.append(nn.BatchNorm1d(os))
            inSize = os
        bns.append(nn.BatchNorm1d(outSize))
        self.actFunc = actFunc()
        self.dropout = nn.Dropout(p=dropout)
        self.hiddens = nn.ModuleList(hiddens)
        self.bns = nn.ModuleList(bns)
        self.out = nn.Linear(inSize, outSize)
        self.bnEveryLayer = bnEveryLayer
        self.dpEveryLayer = dpEveryLayer
        self.outBn = outBn
        self.outAct = outAct
        self.outDp = outDp

    def forward(self, x):
        for h, bn in zip(self.hiddens, self.bns):
            x = h(x)
            if self.bnEveryLayer:
                x = bn(x) if len(x.shape) == 2 else bn(x.transpose(-1, -2)).transpose(-1, -2)
            x = self.actFunc(x)
            if self.dpEveryLayer:
                x = self.dropout(x)
        x = self.out(x)
        if self.outBn: x = self.bns[-1](x) if len(x.shape) == 2 else self.bns[-1](x.transpose(-1, -2)).transpose(-1, -2)
        if self.outAct: x = self.actFunc(x)
        if self.outDp: x = self.dropout(x)
        return x


class BridgeGraph(nn.Module):
    def __init__(self, nodeNum, outSize, device, gcnHiddenSizeList=[], fcHiddenSizeList=[],
                 hdnDropout=0.1, fcDropout=0.2, resnet=True):
        super(BridgeGraph, self).__init__()

        self.nodeEmbedding = TextEmbedding(
            torch.tensor(np.random.normal(size=(max(nodeNum, 0), outSize)), dtype=torch.float32), dropout=hdnDropout,
            name='nodeEmbedding').to(device)  # 桥节点的嵌入向量
        self.nodeNum = nodeNum

        self.device = device

        self.nodeGCN = GCN(outSize, outSize, gcnHiddenSizeList, name='nodeGCN', dropout=hdnDropout, dpEveryLayer=True,
                           outDp=True, bnEveryLayer=True, outBn=True, resnet=resnet).to(device)

        self.fcLinear = MLP(outSize * 2, 1, fcHiddenSizeList, dropout=fcDropout, bnEveryLayer=True, dpEveryLayer=True).to(
            device)

    def forward(self, Drug, CellLine, DrugSim, CellSim, df, cf):
        batch = Drug.shape[0]

        exDrug = Drug.repeat(1, batch, 1)
        exCell = CellLine.repeat(1, batch, 1)

        addDrug = self.remove_vector(exDrug)
        addCell = self.remove_vector(exCell)

        DrugSim = self.remove_value(DrugSim)
        CellSim = self.remove_value(CellSim)

        DrugSim = DrugSim.view(DrugSim.shape[0],DrugSim.shape[2])
        CellSim = CellSim.view(CellSim.shape[0],CellSim.shape[2])

        # print(DrugSim)
        # print(CellSim)

        if self.nodeNum > 0:
            Bnode = self.nodeEmbedding.dropout2(self.nodeEmbedding.dropout1(self.nodeEmbedding.embedding.weight)).repeat(
                len(Drug), 1, 1)
            node = torch.cat([Drug, CellLine, Bnode, addDrug, addCell], dim=1)

            nodeCore = torch.cat([Drug, CellLine, Bnode], dim=1)  # => batchSize × nodeNum × outSize
            nodeDist = torch.sqrt(torch.sum(nodeCore ** 2, dim=2, keepdim=True) + 1e-8)  # => batchSize × nodeNum × 1

            cosNodeCore = torch.matmul(nodeCore, nodeCore.transpose(1, 2)) / (
                    nodeDist * nodeDist.transpose(1, 2) + 1e-8)  # => batchSize × nodeNum × nodeNum
            # print(cosNodeCore)

            cosNode = torch.zeros(batch, node.shape[1], node.shape[1])
            cosNode[:, :2 + self.nodeNum, :2 + self.nodeNum] = cosNodeCore
            # print(cosNode)

            # print(cosNode[:, 0, 2 + self.nodeNum: 2 + self.nodeNum + batch-1].shape)
            # print(cosNode[:, 2 + self.nodeNum: 2 + self.nodeNum + batch-1, 0].shape)
            cosNode[:, 0, 2 + self.nodeNum: 2 + self.nodeNum + batch-1] = DrugSim
            cosNode[:, 1, 2 + self.nodeNum + batch - 1:] = CellSim
            # print(cosNode)

            cosNode[:, 2 + self.nodeNum: 2 + self.nodeNum + batch-1, 0] = DrugSim
            cosNode[:, 2 + self.nodeNum + batch - 1:, 1] = CellSim
            # print(cosNode)

            cosNode[cosNode < 0] = 0
            cosNode[:, range(node.shape[1]), range(node.shape[1])] = 1  # => batchSize × nodeNum × nodeNum
            # if self.maskDTI: cosNode[:, 0, 1] = cosNode[:, 1, 0] = 0
            D = torch.eye(node.shape[1], dtype=torch.float32, device=self.device).repeat(len(Drug), 1,
                                                                                         1)  # => batchSize × nodeNum × nodeNum
            D[:, range(node.shape[1]), range(node.shape[1])] = 1 / (torch.sum(cosNode, dim=2) ** 0.5).to(self.device)
            cosNode = cosNode.to(self.device)
            pL = torch.matmul(torch.matmul(D, cosNode), D)  # => batchSize × batchnodeNum × nodeNumSize
            node_gcned = self.nodeGCN(node, pL)  # => batchSize × nodeNum × outSize

            # print(df.shape)
            # print(node_gcned[:, 0, :].shape)
            d = torch.cat((node_gcned[:, 0, :], df), dim=1)
            # print(d.shape)
            c = torch.cat((node_gcned[:, 1, :], cf), dim=1)

            node_embed = d * c  # => batchSize × outSize
            # print(node_embed.shape)

            return self.fcLinear(node_embed).squeeze(dim=1)

    def remove_vector(self, tensor):
        batch_size = tensor.size(0)
        # 创建掩码，用于标记要删除的向量位置
        mask = torch.eye(batch_size, dtype=bool).unsqueeze(2).repeat(1, 1, tensor.size(2))
        # 使用掩码来选择除了对角线外的所有向量
        result = tensor[~mask].view(batch_size, batch_size - 1, tensor.size(2))

        return result

    def remove_value(self, tensor):
        batch_size = tensor.size(0)
        tensor_size = tensor.size(2)
        # 创建索引，用于选择要删除的值
        indices = torch.arange(tensor_size).unsqueeze(0).repeat(batch_size, 1)
        # 生成一个掩码，将要删除的值设为 True
        mask = indices.unsqueeze(1) != torch.arange(batch_size).unsqueeze(1).unsqueeze(-1)
        # 使用掩码选择要保留的值，并重新调整形状
        result = tensor[mask].view(batch_size, 1, tensor_size - 1)

        return result

    # def batchGraph(self, Drug, CellLine, DrugSim, CellLineSim):

    #     # Drug、CellLine: batchSize × outSize
    #     batch = Drug.shape[0]
    #     if self.nodeNum > 0:
    #         # 每一批数据都生成一个桥节点，表示该批数据Drug-CellLine的联系
    #         node = self.nodeEmbedding.dropout2(
    #             self.nodeEmbedding.dropout1(self.nodeEmbedding.embedding.weight)).repeat(len(Drug), 1)

    #         node = torch.cat([Drug, CellLine, node], dim=0)  # => batchSize × (nodeNum+2) × outSize

    #         # 为计算Cos相似度做准备
    #         nodeDist = torch.sqrt(torch.sum(node ** 2, dim=1, keepdim=True) + 1e-8)  # => batchSize × nodeNum × 1

    #         # 计算节点间的Cos相似度
    #         cosNode = torch.matmul(node, node.transpose(0, 1)) / (
    #                 nodeDist * nodeDist.transpose(0, 1) + 1e-8)  # => batchSize × nodeNum × nodeNum

    #         # 归一化DrugSim和CellLineSim
    #         DrugSim = (DrugSim - DrugSim.mean(dim=1, keepdim=True)) / DrugSim.std(dim=1, keepdim=True)
    #         CellLineSim = (CellLineSim - CellLineSim.mean(dim=1, keepdim=True)) / CellLineSim.std(dim=1, keepdim=True)

    #         # # 归一化cosNode的部分区域
    #         # cosNode[:, 2 * batch:] = (cosNode[:, 2 * batch:] - cosNode[:, 2 * batch:].mean(dim=1, keepdim=True)) / cosNode[:, 2 * batch:].std(dim=1, keepdim=True)
    #         # cosNode[2 * batch:, :2 * batch] = (cosNode[2 * batch:, :2 * batch] - cosNode[2 * batch:, :2 * batch].mean(dim=1, keepdim=True)) / cosNode[2 * batch:, :2 * batch].std(dim=1, keepdim=True)

    #         # 深复制 cosNode
    #         cosNode_copy1 = cosNode[:, 2 * batch:].clone()
    #         cosNode_copy2 = cosNode[2 * batch:, :2 * batch].clone()
    #         cosNode_copy1 = (cosNode_copy1 - cosNode_copy1.mean(dim=1, keepdim=True)) / cosNode_copy1.std(dim=1, keepdim=True)
    #         cosNode_copy2 = (cosNode_copy2 - cosNode_copy2.mean(dim=1, keepdim=True)) / cosNode_copy2.std(dim=1, keepdim=True)

    #         # # 归一化 cosNode 的部分区域
    #         # cosNode_copy[:, 2 * batch:] = (cosNode_copy[:, 2 * batch:] - cosNode_copy[:, 2 * batch:].mean(dim=1, keepdim=True)) / cosNode_copy[:, 2 * batch:].std(dim=1, keepdim=True)
    #         # cosNode_copy[2 * batch:, :2 * batch] = (cosNode_copy[2 * batch:, :2 * batch] - cosNode_copy[2 * batch:, :2 * batch].mean(dim=1, keepdim=True)) / cosNode_copy[2 * batch:, :2 * batch].std(dim=1, keepdim=True)

    #         # 给cosNode的部分区域赋值为DrugSim和CellLineSim
    #         cosNode[:batch, :batch] = DrugSim
    #         cosNode[batch: 2 * batch, batch: 2 * batch] = CellLineSim
    #         cosNode[:, 2 * batch:] = cosNode_copy1
    #         cosNode[2 * batch:, :2 * batch] = cosNode_copy2

    #         # 计算最大值和最小值
    #         min_val = cosNode.min(dim=1, keepdim=True)[0]
    #         max_val = cosNode.max(dim=1, keepdim=True)[0]

    #         # 缩放到 [0, 1] 范围内
    #         cosNode = (cosNode - min_val) / (max_val - min_val)

    #         # 本身的相似度为1
    #         cosNode[range(node.shape[0]), range(node.shape[0])] = 1  # => batchSize × nodeNum × nodeNum

    #         # 归一化整个cosNode
    #         cosNode = (cosNode - cosNode.mean(dim=1, keepdim=True)) / cosNode.std(dim=1, keepdim=True)

    #         # 负相似度置为0
    #         cosNode[cosNode < 0] = 0

    #         D = torch.eye(node.shape[0], dtype=torch.float32,
    #                       device=self.device)  # => batchSize × nodeNum × nodeNum            
    #         D[range(node.shape[0]), range(node.shape[0])] = 1 / (torch.sum(cosNode, dim=1) ** 0.5)
    #         pL = torch.matmul(torch.matmul(D, cosNode), D)  # => batchSize × batchnodeNum × nodeNumSize

    #         node_gcned = self.nodeGCN(node, pL)  # => batchSize × nodeNum × outSize

    #         DrugF = node_gcned[:batch, :]
    #         CellLineF = node_gcned[batch:2 * batch, :]

    #         return DrugF, CellLineF

    # def forward(self, Drug, CellLine, DrugSim, CellLineSim):
    #
    #     self.batchGraph(Drug, CellLine, DrugSim, CellLineSim)
    #
    #     # Drug、CellLine: batchSize × 1 × outSize
    #     if self.nodeNum > 0:
    #         # 每一批数据都生成一个桥节点，表示该批数据Drug-CellLine的联系
    #         node = self.nodeEmbedding.dropout2(self.nodeEmbedding.dropout1(self.nodeEmbedding.embedding.weight)).repeat(
    #             len(Drug), 1, 1)
    #
    #         node = torch.cat([Drug, CellLine, node], dim=1)  # => batchSize × (nodeNum+2) × outSize
    #
    #         # 为计算Cos相似度做准备
    #         nodeDist = torch.sqrt(torch.sum(node ** 2, dim=0, keepdim=True) + 1e-8)  # => batchSize × nodeNum × 1
    #
    #         # 计算节点间的Cos相似度
    #         cosNode = torch.matmul(node, node.transpose(1, 2)) / (
    #                 nodeDist * nodeDist.transpose(1, 2) + 1e-8)  # => batchSize × nodeNum × nodeNum
    #
    #         # 负相似度置为0
    #         cosNode[cosNode < 0] = 0
    #
    #         # 本身的相似度为1
    #         cosNode[:, range(node.shape[1]), range(node.shape[1])] = 1  # => batchSize × nodeNum × nodeNum
    #
    #         D = torch.eye(node.shape[1], dtype=torch.float32, device=self.device)  # => batchSize × nodeNum × nodeNum
    #
    #         D = D.repeat(len(Drug), 1, 1)
    #
    #         D[:, range(node.shape[1]), range(node.shape[1])] = 1 / (torch.sum(cosNode, dim=2) ** 0.5)
    #         pL = torch.matmul(torch.matmul(D, cosNode), D)  # => batchSize × batchnodeNum × nodeNumSize
    #
    #         node_gcned = self.nodeGCN(node, pL)  # => batchSize × nodeNum × outSize
    #
    #         DrugF = node_gcned[:, 0, :]
    #         CellLineF = node_gcned[:, 1, :]
    #
    #         return DrugF, CellLineF


if __name__ == '__main__':
    Bridge = BridgeGraph(nodeNum=3, outSize=2, device='cpu', gcnHiddenSizeList=[128, 128], fcHiddenSizeList=[128])
    Drug = torch.randn(3, 1, 2)
    CellLine = torch.randn(3, 1, 2)

    # Drug = torch.randn(10, 20)
    # CellLine = torch.randn(10, 20)

    DrugSim = torch.randn(3, 1, 3)
    print(DrugSim)
    CellLineSim = torch.randn(3, 1, 3)
    print(CellLineSim)
    score = Bridge(Drug, CellLine, DrugSim, CellLineSim)
    print(score)

    print(1)
