# Script for Testing and Benchmarking Convolutional Layer on GPU
# (See tf.nn.conv2d)
# Important: for correctness checks, please make sure the input dimensions are the same.

from functools import cache
import tensorflow as tf
import time
import sys
import numpy as np

CHECK = False

if __name__ == "__main__":
    batch_size = 256
    width = 56
    height = 56
    channels_in = 64
    channels_out = 64
    kernel_size = 3
    stride = 1

    num_runs = 100
    
    if (len(sys.argv) == 2):
        if (sys.argv[1] == '--check') or (sys.argv[1] == '-c'):
            CHECK = True
            print("Performing correctness check:")
        else:
            raise ValueError("Usage: >python conv2d_gpu_tf.py [--check]")

    if CHECK:
        input_shape = (1,4,4,2)
        filters_shape = (3,3,2,2)
        input_array = np.zeros(input_shape)
        filters_array = np.zeros(filters_shape)

        for i in range(input_shape[0]):
            for j in range(input_shape[1]):
                for k in range(input_shape[2]):
                    for l in range(input_shape[3]):
                        input_array[i,j,k,l] = 0.1 * (i+j+k+l)

        for i in range(filters_shape[0]):
            for j in range(filters_shape[1]):
                for k in range(filters_shape[2]):
                    for l in range(filters_shape[3]):
                        filters_array[i,j,k,l] = 0.1 * (i+j+k+l)
        
        input = tf.convert_to_tensor(input_array)
        filters = tf.convert_to_tensor(filters_array)
        out = tf.nn.conv2d(input, filters, stride, padding="SAME")
        print(out)
    else:
        with tf.device('/GPU:0'):
            best = None
            times = []
            for i in range(num_runs):
                input = tf.random.uniform([batch_size, height, width, channels_in])
                filters = tf.random.uniform([kernel_size, kernel_size, channels_in, channels_out])
                start = time.time()
                out = tf.nn.conv2d(input, filters, stride, padding="SAME")
                end = time.time()
                t = end * 1000 - start * 1000
                times.append(t)
                if not best or t < best: best = t
            avg_time = sum(times) / len(times)
            print('Average: {} ms'.format(avg_time))
            print('Best: {} ms'.format(best))
