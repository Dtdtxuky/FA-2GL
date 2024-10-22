# import numpy as np
# from rdkit import Chem
# from rdkit.Chem import rdMolTransforms

# def AtomType(SMILES):
#     # 解析 SMILES 表达式为分子对象
#     mol = Chem.MolFromSmiles(SMILES)

#     atom_types = [atom.GetAtomicNum() for atom in mol.GetAtoms()]

#     l = len(atom_types)
#     # 获得最多的药物分子的原子数
#     max_len = 96

#     # 填充 max_len - len(atom_types) 个 0
#     atom_types += [0] * (max_len - len(atom_types))

#     # 获得Mask矩阵
#     if l < max_len:
#         input_mask = ([1] * l) + ([0] * (max_len - l))
#     else:
#         input_mask = [1] * max_len

#     atom_types = np.array(atom_types)
#     input_mask = np.array(input_mask)

#     return atom_types, input_mask

# def MolDistance(SMILES):
#     # 解析 SMILES 表达式为分子对象
#     mol = Chem.MolFromSmiles(SMILES)

#     # 计算原子之间的距离矩阵
#     distance_matrix = rdMolTransforms.GetDistanceMatrix(mol)

#     # 获得最多的药物分子的原子数


#     # 填充0


#     return distance_matrix