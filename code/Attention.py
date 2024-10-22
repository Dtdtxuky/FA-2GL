# import torch

# def point_wise_feed_forward_network(d_model, dff):
#     return torch.nn.Sequential(
#         torch.nn.Linear(d_model, dff),
#         torch.nn.ELU(),
#         torch.nn.Linear(dff, d_model)
#     )
