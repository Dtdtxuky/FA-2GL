# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from Attention import point_wise_feed_forward_network
# import torch.nn.functional as F

# class AtomicMultiHeadAttention(torch.nn.Module):
#     def __init__(self, d_model, num_heads, expansion_type='variable', basis_distance=30,
#                  use_attention_scale=True,
#                  use_atom_embedding=False):

#         super(AtomicMultiHeadAttention, self).__init__()
#         self.num_heads = num_heads
#         self.d_model = d_model
#         self.basis_distance = basis_distance

#         assert d_model % self.num_heads == 0
#         self.expansion_type = expansion_type

#         self.depth = d_model // self.num_heads

#         self.wq = nn.Linear(d_model, d_model).to('cuda')
#         self.wk = nn.Linear(d_model, d_model).to('cuda')
#         self.wv = nn.Linear(d_model, d_model).to('cuda')

#         self.dense = nn.Linear(d_model, d_model).to('cuda')

#         # Radial kernel
#         self.radial_expansion_1 = nn.Linear(self.depth, self.depth).to('cuda')
#         self.radial_expansion_2 = nn.Linear(self.depth, self.depth * self.num_heads).to('cuda')

#         self.radial_expansion_sf = nn.Linear(self.depth * self.num_heads, self.depth * self.num_heads).to('cuda')

#         # Polynomial kernel
#         self.polynomial_expansion = nn.Linear(self.depth, self.depth).to('cuda')

#         self.schnet_1 = nn.Linear(64, 64).to('cuda')
#         self.schnet_2 = nn.Linear(self.depth * self.num_heads, self.depth * self.num_heads).to('cuda')

#         # Attention Scale
#         self.use_attention_scale = use_attention_scale
#         if self.use_attention_scale:
#             self.attn_scale_weight = nn.Parameter(torch.tensor(0.1))

#         # Atom Embedding
#         self.use_atom_embedding = use_atom_embedding
#         if self.use_atom_embedding:
#             self.atom_embedding = nn.Embedding(200, 64)


#     def expand_to_radial(self, distance, atom_type, expansion_type='variable'):
#         '''

#         :param distance: distance matrix, [seq_len, seq_len]
#         :param expansion_type: string, expansion type.
#         :return: tensor, [seq_len, seq_len, self.depth]
#         '''

#         if expansion_type == 'physnet':
#             exp_distance = distance.unsqueeze(-1)
#             divident = torch.arange(0, 30, 0.1)
#             rbf = exp_distance / divident
#             rbf = (1 - rbf ** 6 + 15 * rbf ** 4 - 10 * rbf ** 2) * (rbf <= 1).float()  # Apply maximum constraint
#             exp_distance = self.radial_expansion_1(rbf)
#             exp_distance = self.radial_expansion_2(exp_distance)

#             return exp_distance

#         if expansion_type == 'schnet':
#             exp_distance = distance.unsqueeze(-1)
#             dist_delta = torch.arange(0, self.basis_distance, 0.1)
#             exp_distance = exp_distance - dist_delta
#             exp_distance = exp_distance.pow(2) * (-10)
#             exp_distance = exp_distance.exp()

#             if self.use_atom_embedding:
#                 atom_embedding = self.atom_embedding(atom_type)

#                 seq_len = exp_distance.size(1)

#                 a_embedding = atom_embedding.unsqueeze(1).expand(-1, seq_len, -1)
#                 b_embedding = atom_embedding.unsqueeze(2).expand(-1, -1, seq_len)

#                 exp_distance = torch.cat([exp_distance, a_embedding + b_embedding], dim=3)

#             exp_distance = self.schnet_1(exp_distance)
#             exp_distance = self.schnet_2(exp_distance)

#             return exp_distance

#         if expansion_type == 'bessel':
#             exp_distance = distance.unsqueeze(-1)
#             exp_distance = self.bessel(exp_distance)
#             exp_distance = exp_distance.squeeze(-1)
#             exp_distance = exp_distance.permute(0, 2, 3, 1)
#             exp_distance = self.schnet_1(exp_distance)
#             exp_distance = self.schnet_2(exp_distance)

#             return exp_distance

#         if expansion_type == 'polynomial2':
#             inv_dist = torch.minimum(1 / distance, 3) / 3
#             poly_dist = torch.stack([inv_dist, inv_dist ** 2, inv_dist ** 3], dim=-1)
#             poly_dist = self.polynomial_expansion(poly_dist)

#             exp_distance = distance.unsqueeze(-1)
#             exponent_expansion = torch.matmul(exp_distance * -1,
#                                               torch.tensor([list(range(1, self.depth + 1))], dtype=torch.float32))
#             exponent_expansion = torch.exp(exponent_expansion) * poly_dist

#             exponent_expansion = self.radial_expansion_sf(exponent_expansion)
#             exponent_expansion = self.radial_expansion_2(exponent_expansion)

#             return exponent_expansion

#         if expansion_type == 'polynomial':
#             inv_dist = torch.minimum(1 / distance, 3) / 3
#             poly_dist = torch.stack([inv_dist, inv_dist ** 2, inv_dist ** 3], dim=-1)
#             poly_dist = self.polynomial_expansion(poly_dist)

#             exp_distance = distance.unsqueeze(-1)
#             exponent_expansion = torch.matmul(exp_distance * -1,
#                                               torch.tensor([list(range(1, self.depth + 1))], dtype=torch.float32))
#             exponent_expansion = torch.exp(exponent_expansion) * poly_dist

#             exponent_expansion = self.radial_expansion_1(exponent_expansion)
#             exponent_expansion = self.radial_expansion_2(exponent_expansion)

#             return exponent_expansion




#         if expansion_type == 'variable':
#             distance = distance.unsqueeze(-1).float()
#             range_tensor = torch.arange(0.1, 3.001, 2.9 / (self.depth - 1), dtype=torch.float32)
#             range_tensor = range_tensor.view(1, 1, self.depth).to('cuda')

#             exponent_expansion = torch.matmul(distance * -1, range_tensor).to('cuda')
#             exponent_expansion = torch.clamp(0.3 / distance, max=3) * torch.exp(exponent_expansion)

#             exponent_expansion = self.radial_expansion_1(exponent_expansion)
#             exponent_expansion = self.radial_expansion_2(exponent_expansion)

#             return exponent_expansion

#         if expansion_type == 'fixed':
#             distance = distance.unsqueeze(-1)
#             exponent_expansion = torch.matmul(distance * -1,
#                                               torch.tensor([list(range(1, self.depth + 1))], dtype=torch.float32))
#             return torch.minimum(0.5 / distance, 2) * torch.exp(exponent_expansion)

#         if expansion_type == 'linear':
#             distance = distance.unsqueeze(-1)
#             exponent_expansion = torch.matmul(distance * -1,
#                                               torch.tensor([[1.0 for i in range(self.depth * self.num_heads)]],
#                                                            dtype=torch.float32))
#             return torch.minimum(0.5 / distance, 2) * torch.exp(exponent_expansion)


#     def split_heads(self, x, batch_size):
#         """划分注意力头"""
#         x = x.view(batch_size, -1, self.num_heads, self.depth)
#         return x.permute(0, 2, 1, 3)


#     def scaled_weighted_dot_product_attention(self, q, k, v, weight, mask, attention_scale):
#         q = q.unsqueeze(-2).to('cuda')  # (batch, head, seq_len, 1, depth)
#         k = k.unsqueeze(-3).to('cuda')  # (batch, head, 1, seq_len, depth)

#         r = weight
#         dk = k.size(-1)

#         scaled_attention_logits = torch.sum(q * k * r, dim=-1) / torch.sqrt(torch.tensor(dk).float())

#         USE_SOFTMAX = True
#         # Add mask to scaled tensor
#         if mask is not None:
#             if USE_SOFTMAX:
#                 scaled_attention_logits += (mask * -1e9)
#             else:
#                 scaled_attention_logits = scaled_attention_logits * (1 - mask)

#         # Softmax normalization
#         attention_weights = scaled_attention_logits
#         if USE_SOFTMAX:
#             attention_weights = torch.nn.functional.softmax(attention_weights, dim=-1)

#         if attention_scale is not None:
#             mask = mask.float()
#             attention_weights_mean = torch.sum(attention_weights, dim=-1, keepdim=True)
#             mask_mean = torch.mean(1 - mask, dim=-1, keepdim=True)

#             seq_len = attention_weights.shape[-1]
#             attention_weights_mean = attention_weights_mean * (1 - mask) / seq_len / mask_mean

#             attention_weights = (attention_weights - attention_weights_mean) * (
#                         1 + attention_scale) + attention_weights_mean

#         output = torch.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

#         return output, attention_weights


#     def forward(self,vkq_weight, mask, training):

#         q, k, v, weight, atom_type = vkq_weight

#         q = q.to('cuda')
#         k = k.to('cuda')
#         v = v.to('cuda')
#         weight = weight.to('cuda')
#         atom_type = atom_type.to('cuda')

#         batch_size = q.size(0)

#         q = self.wq(q).to('cuda')  # (batch_size, seq_len, d_model)
#         k = self.wk(k).to('cuda')  # (batch_size, seq_len, d_model)
#         v = self.wv(v).to('cuda')  # (batch_size, seq_len, d_model)

#         q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
#         k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
#         v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
#         # print(16)

#         shifted_distance = self.expand_to_radial(weight, atom_type, expansion_type=self.expansion_type)
#         # print(17)
#         shifted_distance = shifted_distance.view(batch_size, shifted_distance.size(1), shifted_distance.size(2),
#                                                  self.num_heads, self.depth)
#         # print(18)
#         shifted_distance = shifted_distance.permute(0, 3, 1, 2, 4)
#         # print(19)
#         scaled_attention, attention_weights = self.scaled_weighted_dot_product_attention(q, k, v, shifted_distance,
#                                                                                          mask, self.attn_scale_weight)
#         # print(20)
#         scaled_attention = scaled_attention.permute(0, 2, 1, 3)

#         # print(21)
#         concat_attention = scaled_attention.reshape(batch_size, -1, self.d_model)

#         # print(22)
#         output = self.dense(concat_attention)
#         # print(23)
#         return output, attention_weights

#     # def expand_to_radial(self, distance, atom_type, expansion_type='variable'):
#     #     '''
#     #
#     #     :param distance: distance matrix, [seq_len, seq_len]
#     #     :param expansion_type: string, expansion type.
#     #     :return: tensor, [seq_len, seq_len, self.depth]
#     #     '''
#     #
#     #     if expansion_type == 'physnet':
#     #         exp_distance = tf.expand_dims(distance, axis=-1)
#     #         divident = [np.arange(0, 30, 0.1).tolist()]
#     #         rbf = exp_distance / divident
#     #         rbf = (1 - rbf * rbf * rbf * 6 + 15 * rbf * rbf * rbf * rbf - 10 * rbf * rbf * rbf * rbf * rbf)
#     #         rbf = tf.maximum(rbf, 0.0)
#     #         exp_distance = self.radial_expansion_1(rbf)
#     #         exp_distance = self.radial_expansion_2(exp_distance)
#     #
#     #         return exp_distance
#     #
#     #     if expansion_type == 'schnet':
#     #         exp_distance = tf.expand_dims(distance, axis=-1)
#     #         dist_delta = [np.arange(0, self.basis_distance, 0.1).tolist()]
#     #         exp_distance = exp_distance - dist_delta
#     #         exp_distance = exp_distance * exp_distance * (-10)
#     #         exp_distance = tf.exp(exp_distance)
#     #
#     #         if self.use_atom_embedding:
#     #             atom_embedding = self.atom_embedding(atom_type)
#     #
#     #             seq_len = tf.shape(exp_distance)[1]
#     #
#     #             a_embedding = tf.repeat(
#     #                 tf.expand_dims(atom_embedding, axis=1),
#     #                 repeats=seq_len,
#     #                 axis=1
#     #             )
#     #             b_embedding = tf.repeat(
#     #                 tf.expand_dims(atom_embedding, axis=2),
#     #                 repeats=seq_len,
#     #                 axis=2
#     #             )
#     #
#     #             exp_distance = tf.concat([
#     #                 exp_distance, a_embedding + b_embedding
#     #             ], 3)
#     #
#     #         exp_distance = self.schnet_1(exp_distance)
#     #         exp_distance = self.schnet_2(exp_distance)
#     #
#     #         exp_distance = exp_distance
#     #         return exp_distance
#     #
#     #     if expansion_type == 'bessel':
#     #         exp_distance = tf.expand_dims(distance, axis=-1)
#     #         exp_distance = self.bessel(exp_distance)
#     #         exp_distance = tf.squeeze(exp_distance, axis=-1)
#     #         exp_distance = tf.transpose(exp_distance, [0, 2, 3, 1])
#     #         exp_distance = self.schnet_1(exp_distance)
#     #         exp_distance = self.schnet_2(exp_distance)
#     #
#     #         return exp_distance
#     #
#     #     if expansion_type == 'polynomial2':
#     #         inv_dist = tf.minimum(1 / distance, 3) / 3
#     #         poly_dist = tf.stack([inv_dist, inv_dist * inv_dist, inv_dist * inv_dist * inv_dist], axis=-1)
#     #         poly_dist = self.polynomial_expansion(poly_dist)
#     #
#     #         exp_distance = tf.expand_dims(distance, axis=-1)
#     #         exponent_expansion = tf.matmul(exp_distance * -1, [np.arange(1, 3.001, 2.0 / (self.depth - 1)).tolist()])
#     #         exponent_expansion = tf.exp(exponent_expansion) * poly_dist
#     #
#     #         exponent_expansion = self.radial_expansion_sf(exponent_expansion)
#     #         exponent_expansion = self.radial_expansion_2(exponent_expansion)
#     #
#     #         return exponent_expansion
#     #
#     #     if expansion_type == 'polynomial':
#     #         inv_dist = tf.minimum(1 / distance, 3) / 3
#     #         poly_dist = tf.stack([inv_dist, inv_dist * inv_dist, inv_dist * inv_dist * inv_dist], axis=-1)
#     #         poly_dist = self.polynomial_expansion(poly_dist)
#     #
#     #         exp_distance = tf.expand_dims(distance, axis=-1)
#     #         exponent_expansion = tf.matmul(exp_distance * -1, [np.arange(1, 3.001, 2.0 / (self.depth - 1)).tolist()])
#     #         exponent_expansion = tf.exp(exponent_expansion) * poly_dist
#     #
#     #         exponent_expansion = self.radial_expansion_1(exponent_expansion)
#     #         exponent_expansion = self.radial_expansion_2(exponent_expansion)
#     #
#     #         return exponent_expansion
#     #
#     #     if expansion_type == 'variable':
#     #         distance = tf.expand_dims(distance, axis=-1)
#     #         exponent_expansion = tf.matmul(distance * -1, [np.arange(0.1, 3.001, 2.9 / (self.depth - 1)).tolist()])
#     #         exponent_expansion = tf.minimum(0.3 / distance, 3) * tf.exp(exponent_expansion)
#     #         exponent_expansion = self.radial_expansion_1(exponent_expansion)
#     #         exponent_expansion = self.radial_expansion_2(exponent_expansion)
#     #
#     #         return exponent_expansion
#     #
#     #     if expansion_type == 'fixed':
#     #         distance = tf.expand_dims(distance, axis=-1)
#     #         exponent_expansion = tf.matmul(distance * -1, [np.arange(0.1, 3.001, 2.9 / (self.depth - 1)).tolist()])
#     #         return tf.minimum(0.5 / distance, 2) * tf.exp(exponent_expansion)
#     #
#     #     if expansion_type == 'linear':
#     #         distance = tf.expand_dims(distance, axis=-1)
#     #         exponent_expansion = tf.matmul(distance * -1, [[1.0 for i in range(self.depth * self.num_heads)]])
#     #         return tf.minimum(0.5 / distance, 2) * tf.exp(exponent_expansion)
#     #
#     #
#     # def split_heads(self, x, batch_size):
#     #     """Split the last dimension into (num_heads, depth).
#     #     Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
#     #     """
#     #     x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
#     #     return tf.transpose(x, perm=[0, 2, 1, 3])
#     #
#     # def call(self, vkq_weight, mask=None, training=None):
#     #     v, k, q, weight, atom_type = vkq_weight
#     #     batch_size = tf.shape(q)[0]
#     #
#     #     q = self.wq(q)  # (batch_size, seq_len, d_model)
#     #     k = self.wk(k)  # (batch_size, seq_len, d_model)
#     #     v = self.wv(v)  # (batch_size, seq_len, d_model)
#     #
#     #     q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
#     #     k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
#     #     v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
#     #
#     #     shifted_distance = self.expand_to_radial(weight, atom_type, expansion_type=self.expansion_type)
#     #
#     #
#     #     shifted_distance = tf.reshape(shifted_distance, (batch_size, tf.shape(shifted_distance)[1], tf.shape(shifted_distance)[2], self.num_heads, self.depth))
#     #     shifted_distance = tf.transpose(shifted_distance, [0, 3, 1, 2, 4])
#     #
#     #     scaled_attention, attention_weights = scaled_weighted_dot_product_attention(
#     #         q, k, v, shifted_distance, mask, self.attn_scale_weight)
#     #
#     #     scaled_attention = tf.transpose(scaled_attention,
#     #                                     perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
#     #
#     #     concat_attention = tf.reshape(scaled_attention,
#     #                                   (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
#     #
#     #     output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
#     #
#     #     return output, attention_weights


# class AtomicLayer(torch.nn.Module):
#     def __init__(self, d_model, num_heads, dff, rate=0.0, basis_distance=30, expansion_type='variable',
#                  use_attention_scale=True,
#                  use_atom_embedding=False,
#                  use_parallel_mlp=False):
#         super(AtomicLayer, self).__init__()

#         self.AMHA = AtomicMultiHeadAttention(d_model, num_heads, expansion_type=expansion_type, basis_distance=basis_distance,
#             use_attention_scale=use_attention_scale,
#             use_atom_embedding=use_atom_embedding
#         )

#         self.ffn = point_wise_feed_forward_network(d_model, dff).to('cuda')
#         self.layernorm1 = torch.nn.LayerNorm(d_model, eps=1e-6).to('cuda')
#         if not use_parallel_mlp:
#             self.layernorm2 = torch.nn.LayerNorm(d_model, eps=1e-6).to('cuda')

#         self.dropout1 = F.dropout(p=0.5)
#         self.dropout2 = F.dropout(p=0.5)

#         self._save_recent_coeff = None
#         print(
#             f"use_attention_scale {use_attention_scale} | use_atom_embedding {use_atom_embedding} | use_parallel_mlp {use_parallel_mlp}")
#         self.use_parallel_mlp = use_parallel_mlp


#     def forward(self,state_weight, mask, training=None):
#         # 原子类型编码，距离矩阵，原子类型
#         orbital_state, overlap_weight, atom_type = state_weight

#         orbital_state = orbital_state.to('cuda')
#         # print(14)
#         attn_output, tmp_attn = self.AMHA([orbital_state, orbital_state, orbital_state, overlap_weight, atom_type],
#                                              mask, training)
#         attn_output = attn_output.to('cuda')                           

#         self._save_recent_coeff = tmp_attn.to('cuda')
#         # print(15)
#         if self.use_parallel_mlp:
#             attn_output = self.dropout1(attn_output).to('cuda')
#             ffn_output = self.ffn(orbital_state)  # (batch_size, input_seq_len, d_model)
#             ffn_output = self.dropout2(ffn_output)

#             out1 = self.layernorm1(orbital_state + attn_output + ffn_output)  # (batch_size, input_seq_len, d_model)
#             return out1

#         else:
#             attn_output = self.dropout1(attn_output).to('cuda')
#             out1 = self.layernorm1(orbital_state + attn_output).to('cuda')  # (batch_size, input_seq_len, d_model)
#             ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
#             ffn_output = self.dropout2(ffn_output)
#             new_state = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

#             return new_state

