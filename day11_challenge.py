import torch

batch_size=2
sequence_length=4
d_k=8

Q = torch.randn(size=(batch_size, sequence_length, d_k))

K = torch.randn(size=(batch_size, sequence_length, d_k))

K_transposed = K.transpose(-2,-1)

matrix_multiply = torch.matmul(Q,K_transposed)
scaled_attention_scores = matrix_multiply/(d_k**0.5)

print(f"shape of Q: {Q.shape}")
print(f"shape of Transposed K: {K_transposed.shape}")
print(f"Attention score matrix: {scaled_attention_scores}")

weights = torch.softmax(scaled_attention_scores, dim=-1)
print(weights)