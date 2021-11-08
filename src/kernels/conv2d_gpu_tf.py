# Convolutional Layer on GPU
# (See tf.nn.conv2d)

import tensorflow as tf
import time


if __name__ == "__main__":
    batch_size = 8
    width = 120
    height = 100
    channels_in = 3
    channles_out = 3
    kernel_size = 5
    stride = 1
    
    num_runs = 100
    
    with tf.device('/GPU:0'):

        input = tf.random.uniform([batch_size, height, width, channels_in])
        filters = tf.random.uniform([kernel_size, kernel_size, channels_in, channels_out])
        
        best = None
        times = []
        for i in range(num_runs):
            start = time.time()
            out = tf.nn.conv2d(input, filters, stride, padding="SAME")
            end = time.time()
            t = (end - start) / num_iter
            times.append(t)
            if not best or t < best: best = t
        avg_time = sum(times) / len(times)
        print('Average: {} ms'.format(1000 * avg_time))
        print('Best: {} ms'.format(1000 * best))
