# import torch
# from Atomic import *
#
# class InferenceAtomicNet(torch.nn.Module):
#     def __init__(self,chemical_sequence_length,
#                  chemical_vocab_size,
#                  transformer_model_dim,
#                  transformer_num_heads,
#                  transformer_hidden_dimension,
#                  transformer_num_layers,
#                  label_count=12,
#                  basis_distance=30,
#                  is_regression=False,
#                  use_attention_scale = True,
#                  use_atom_embedding = False,
#                  use_parallel_mlp = False):
#
#         super(InferenceAtomicNet, self).__init__()
#
#         # Transformer 编码的层数
#         self.layer_num = transformer_num_layers
#
#         # 原子类型To嵌入编码层
#         self.embedding = torch.nn.Embedding(chemical_vocab_size, transformer_model_dim)
#
#         # 原子GTN编码层（共self.layer_num层）
#         self.atomic_layer = [
#             AtomicLayer(transformer_model_dim,
#                         transformer_num_heads,
#                         transformer_hidden_dimension,
#                         basis_distance=basis_distance,
#                         expansion_type='variable',
#                         use_attention_scale=use_attention_scale,
#                         use_atom_embedding=use_atom_embedding,
#                         use_parallel_mlp=use_parallel_mlp
#                         ) for i in range(transformer_num_layers)
#         ]
#
#         # 层归一化
#         self.norm1 = nn.LayerNorm([transformer_model_dim])
#         self.norm2 = nn.LayerNorm([transformer_model_dim])
#         self.norm3 = nn.LayerNorm([transformer_model_dim])
#         self.LayerList = [self.norm1, self.norm2, self.norm3]
#
#     def forward(self, Drug):
#
#         AtomType = Drug[0].to('cuda')
#         AtomDis = Drug[2]
#         DrugMask = Drug[1]
#
#         # 掩码矩阵
#         DrugMask = DrugMask.unsqueeze(1).unsqueeze(2)
#         DrugMask = (1.0 - DrugMask).to('cuda')
#
#         orbit_state = self.embedding(AtomType)
#
#         for i in range(self.layer_num):
#             # print(12)
#             orbit_state = self.LayerList[i](orbit_state)
#             # print(13)
#             orbit_state = self.atomic_layer[i]([orbit_state, AtomDis, AtomType], DrugMask)
#
#         return orbit_state
#
#     def GetEmbeding(self,Drug):
#         AtomType = Drug[0].to('cuda')
#         AtomDis = Drug[2]
#         DrugMask = Drug[1]
#
#         # 掩码矩阵
#         DrugMask = DrugMask.unsqueeze(1).unsqueeze(2)
#         DrugMask = (1.0 - DrugMask).to('cuda')
#
#         orbit_state = self.embedding(AtomType)
#
#         return orbit_state
#
#
#
#
#
#
