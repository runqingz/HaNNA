# Script for Testing and Benchmarking Convolutional Layer on GPU
# (See tf.nn.conv2d)

from functools import cache
import tensorflow as tf
import time
import sys
import numpy as np

CHECK = False

if __name__ == "__main__":
    #Make sure this is matching HaNNA implementation
    batch_size = 8
    width = 256
    height = 256
    channels_in = 128

    num_runs = 100

    if (len(sys.argv) == 2):
        if (sys.argv[1] == '--check') or (sys.argv[1] == '-c'):
            CHECK = True
            print("Performing correctness check:")
        else:
            raise ValueError("Usage: >python relu_tf.py [--check]")

    if CHECK:
        input_shape = (1,4,4,2)
        input_array = np.zeros(input_shape)

        for i in range(input_shape[0]):
            for j in range(input_shape[1]):
                for k in range(input_shape[2]):
                    for l in range(input_shape[3]):
                        input_array[i,j,k,l] = 0.1 * (i+j+k+l)

        input = tf.convert_to_tensor(input_array)
        out = tf.nn.relu(input)
        print(out)
    else:
        with tf.device('/GPU:0'):
            input = tf.random.uniform([batch_size, height, width, channels_in])
            best = None
            times = []
            
            for i in range(num_runs):
                start = time.time()
                out = tf.nn.relu(input)
                end = time.time()
                t = end - start
                times.append(t)
                if not best or t < best: best = t
            avg_time = sum(times) / len(times)
            print('Average: {} ms'.format(1000 * avg_time))
            print('Best: {} ms'.format(1000 * best))