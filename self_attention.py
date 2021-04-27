
import torch

x = [
  [0, 1, 1], # Input 1
  [1, 1, 2], # Input 2
  [1, 0, 1]  # Input 3
 ]
x = torch.tensor(x, dtype=torch.float32)

w_key = [
  [2, 0, 1],
  [1, 1, 2],
  [1, 1, 1]
]
w_query = [
  [1, 1, 1],
  [0, 1, 1],
  [1, 0, 0]
]
w_value = [
  [2, 0, 0],
  [1, 0, 1],
  [1, 0, 2]
]
w_key = torch.tensor(w_key, dtype=torch.float32)
w_query = torch.tensor(w_query, dtype=torch.float32)
w_value = torch.tensor(w_value, dtype=torch.float32)

keys = x @ w_key
querys = x @ w_query
values = x @ w_value

print(keys)
# tensor([[2., 2., 3.],
#        [5., 3., 5.],
#        [3., 1., 2.]])

print(querys)
# tensor([[1., 1., 1.],
#        [3., 2., 2.],
#       [2., 1., 1.]])

print(values)
# tensor([[2., 0., 3.],
#        [5., 0., 5.],
#        [3., 0., 2.]])

attn_scores = querys @ keys.T

print(attn_scores)
# tensor([[ 7., 13.,  6.],
#         [16., 31., 15.],
#         [ 9., 18.,  9.]])

from torch.nn.functional import softmax
attn_scores_softmax = softmax(attn_scores, dim=-1)
print(attn_scores_softmax)

# tensor([[2.4704e-03, 9.9662e-01, 9.0880e-04],
#         [3.0590e-07, 1.0000e+00, 1.1254e-07],
#         [1.2338e-04, 9.9975e-01, 1.2338e-04]])

# For readability, approximate the above as follows
attn_scores_softmax = [
  [0.0, 0.9, 0.0],
  [0.0, 1.0, 0.0],
  [0.0, 0.9, 0.0]
]

attn_scores_softmax = torch.tensor(attn_scores_softmax)

weighted_values = values[:,None] * attn_scores_softmax.T[:,:,None]

outputs = weighted_values.sum(dim=0)

print(outputs)

# tensor([[4.5000, 0.0000, 4.5000],
#         [5.0000, 0.0000, 5.0000],
#         [4.5000, 0.0000, 4.5000]])
