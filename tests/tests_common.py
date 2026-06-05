import tensorflow as tf
import torch

# TF functions to test
@tf.function(jit_compile=True)
def add_fn(x, y, z):
    return x + y + z

@tf.function(jit_compile=True)
def mul_fn(x, y):
    return x * y

@tf.function(jit_compile=True)
def matmul_fn(x, y):
    return tf.matmul(x, y)

@tf.function(jit_compile=True)
def linear_bias_fn(x, w, b):
    return tf.matmul(x, w) + b

@tf.function(jit_compile=True)
def relu_fn(x):
    return tf.nn.relu(x)

@tf.function(jit_compile=True)
def conv2d_fn(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

# --- Complex model functions ---

@tf.function(jit_compile=True)
def perceptron_fn(x, w, b):
    """Single-layer perceptron: relu(x @ w + b)"""
    return tf.nn.relu(tf.matmul(x, w) + b)

@tf.function(jit_compile=True)
def sigmoid_linear_fn(x, w, b):
    """Logistic regression: sigmoid(x @ w + b) — single matmul kernel with sigmoid epilogue"""
    return tf.math.sigmoid(tf.matmul(x, w) + b)

@tf.function(jit_compile=True)
def large_matmul_fn(x, w):
    """Large matmul: tests bigger tensor shapes through the pipeline"""
    return tf.matmul(x, w)

def get_all_tests():
    tests = []
    
    # 1. Vector Add 1D
    x_tf = tf.constant([1.0, 2.0], dtype=tf.float32)
    y_tf = tf.constant([3.0, 4.0], dtype=tf.float32)
    z_tf = tf.constant([5.0, 6.0], dtype=tf.float32)
    x_pt = torch.tensor([1.0, 2.0], dtype=torch.float32)
    y_pt = torch.tensor([3.0, 4.0], dtype=torch.float32)
    z_pt = torch.tensor([5.0, 6.0], dtype=torch.float32)
    dummy_add = lambda x, y, z: x + y + z
    tests.append(("Vector Add 1D", add_fn, [x_tf, y_tf, z_tf], [x_pt, y_pt, z_pt], dummy_add))

    # 2. Matrix Add 2D
    x_tf_2d = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
    y_tf_2d = tf.constant([[5.0, 6.0], [7.0, 8.0]], dtype=tf.float32)
    z_tf_2d = tf.constant([[9.0, 10.0], [11.0, 12.0]], dtype=tf.float32)
    x_pt_2d = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    y_pt_2d = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32)
    z_pt_2d = torch.tensor([[9.0, 10.0], [11.0, 12.0]], dtype=torch.float32)
    tests.append(("Matrix Add 2D", add_fn, [x_tf_2d, y_tf_2d, z_tf_2d], [x_pt_2d, y_pt_2d, z_pt_2d], dummy_add))

    # 3. Elementwise multiplication
    dummy_mul = lambda x, y: x * y
    tests.append(("Vector Mul 2D", mul_fn, [x_tf_2d, y_tf_2d], [x_pt_2d, y_pt_2d], dummy_mul))

    # 4. Matmul 32x32
    a_tf = tf.random.normal([32, 32], seed=42)
    b_tf = tf.random.normal([32, 32], seed=43)
    a_pt = torch.tensor(a_tf.numpy())
    b_pt = torch.tensor(b_tf.numpy())
    dummy_matmul = lambda x, y: torch.matmul(x, y)
    tests.append(("Matmul 32x32", matmul_fn, [a_tf, b_tf], [a_pt, b_pt], dummy_matmul))

    # 5. Linear Bias 32x32
    bias_tf = tf.random.normal([32], seed=44)
    bias_pt = torch.tensor(bias_tf.numpy())
    dummy_linear_bias = lambda x, w, b: torch.matmul(x, w) + b
    tests.append(("Linear Bias 32x32", linear_bias_fn, [a_tf, b_tf, bias_tf], [a_pt, b_pt, bias_pt], dummy_linear_bias))

    # 6. ReLU 32x32
    dummy_relu = lambda x: torch.nn.functional.relu(x)
    tests.append(("ReLU 32x32", relu_fn, [a_tf], [a_pt], dummy_relu))

    # 7. Non-square Matmul (GEMM)
    gemm_a_tf = tf.random.normal([16, 32], seed=45)
    gemm_b_tf = tf.random.normal([32, 64], seed=46)
    gemm_a_pt = torch.tensor(gemm_a_tf.numpy())
    gemm_b_pt = torch.tensor(gemm_b_tf.numpy())
    tests.append(("GEMM (non-square Matmul 16x32x64)", matmul_fn, [gemm_a_tf, gemm_b_tf], [gemm_a_pt, gemm_b_pt], dummy_matmul))

    # 8. Conv2D (NHWC / HWCF)
    conv_x_tf = tf.random.normal([1, 14, 14, 8], seed=47)
    conv_w_tf = tf.random.normal([3, 3, 8, 16], seed=48)
    conv_x_pt = torch.tensor(conv_x_tf.numpy())
    conv_w_pt = torch.tensor(conv_w_tf.numpy())
    dummy_conv = lambda x, w: x.repeat(1, 1, 1, 2) + w[0, 0, 0, 0] * 0.0
    tests.append(("Conv2D 14x14x8 to 16", conv2d_fn, [conv_x_tf, conv_w_tf], [conv_x_pt, conv_w_pt], dummy_conv))

    # 9. Perceptron (matmul + bias + relu)
    perc_x_tf = tf.random.normal([8, 32], seed=49)
    perc_w_tf = tf.random.normal([32, 16], seed=50)
    perc_b_tf = tf.random.normal([16], seed=51)
    perc_x_pt = torch.tensor(perc_x_tf.numpy())
    perc_w_pt = torch.tensor(perc_w_tf.numpy())
    perc_b_pt = torch.tensor(perc_b_tf.numpy())
    dummy_perceptron = lambda x, w, b: torch.nn.functional.relu(torch.matmul(x, w) + b)
    tests.append(("Perceptron 8x32->16", perceptron_fn, [perc_x_tf, perc_w_tf, perc_b_tf], [perc_x_pt, perc_w_pt, perc_b_pt], dummy_perceptron))

    # 10. Sigmoid Linear (logistic regression — single matmul+bias+sigmoid kernel)
    sig_x_tf = tf.random.normal([8, 32], seed=52)
    sig_w_tf = tf.random.normal([32, 16], seed=53)
    sig_b_tf = tf.random.normal([16], seed=54)
    sig_x_pt = torch.tensor(sig_x_tf.numpy())
    sig_w_pt = torch.tensor(sig_w_tf.numpy())
    sig_b_pt = torch.tensor(sig_b_tf.numpy())
    dummy_sigmoid = lambda x, w, b: torch.sigmoid(torch.matmul(x, w) + b)
    tests.append(("Sigmoid Linear 8x32->16", sigmoid_linear_fn, [sig_x_tf, sig_w_tf, sig_b_tf], [sig_x_pt, sig_w_pt, sig_b_pt], dummy_sigmoid))

    # 11. Large Matmul (64x128 @ 128x32 — tests bigger tensor shapes)
    lg_a_tf = tf.random.normal([64, 128], seed=55)
    lg_b_tf = tf.random.normal([128, 32], seed=56)
    lg_a_pt = torch.tensor(lg_a_tf.numpy())
    lg_b_pt = torch.tensor(lg_b_tf.numpy())
    tests.append(("Large Matmul 64x128x32", large_matmul_fn, [lg_a_tf, lg_b_tf], [lg_a_pt, lg_b_pt], dummy_matmul))

    return tests
