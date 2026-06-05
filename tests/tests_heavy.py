import torch
import torch.nn.functional as F
import tensorflow as tf

def get_heavy_tests():
    # 1. MLP 2-Layer
    # We need a TF fn and a PyTorch dummy fn.
    # TF fn:
    @tf.function(jit_compile=True)
    def mlp_tf(x, w1, b1, w2, b2):
        x1 = tf.matmul(x, w1) + b1
        r1 = tf.nn.relu(x1)
        x2 = tf.matmul(r1, w2) + b2
        r2 = tf.nn.relu(x2)
        return r2
        
    dummy_mlp = lambda x, w1, b1, w2, b2: torch.relu(torch.matmul(torch.relu(torch.matmul(x, w1) + b1), w2) + b2)
    inputs_mlp_torch = [
        torch.randn(8, 32),
        torch.randn(32, 64),
        torch.randn(64),
        torch.randn(64, 16),
        torch.randn(16)
    ]
    inputs_mlp_tf = [tf.convert_to_tensor(t.numpy()) for t in inputs_mlp_torch]
    
    return [
        ('MLP 2-Layer 8x32->64->16', mlp_tf, inputs_mlp_tf, inputs_mlp_torch, dummy_mlp)
    ]
