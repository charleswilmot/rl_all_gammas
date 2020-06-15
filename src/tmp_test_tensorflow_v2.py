import tensorflow as tf
from time import time
import numpy as np


N = 1000
T = 50

@tf.function
def big_computation(inp, weights, add_one=False):
    print("Tracing with parameter add_one = {}".format(add_one))
    b = tf.reduce_sum(tf.matmul(weights, inp))
    if add_one:
        return b + 1
    return b


@tf.function
def one_call_only(inp, weights):
    return big_computation(inp, weights)


@tf.function
def two_calls(inp, weights):
    return big_computation(inp, weights) + big_computation(inp, weights, add_one=True)


print("\n" * 4)
weights = np.random.normal(size=(N, N))

one_call_only(weights, weights)
two_calls(weights, weights)

t0 = time()
for i in range(T):
    inp = np.random.normal(size=(N, N))
    one_call_only(inp, weights)
t1 = time()
print("\n" * 4)
print("One call  took in average {: 5d} us".format(int((t1 - t0) / T * 1000000)))


t0 = time()
for i in range(T):
    inp = np.random.normal(size=(N, N))
    two_calls(inp, weights)
t1 = time()
print("Two calls took in average {: 5d} us".format(int((t1 - t0) / T * 1000000)))
