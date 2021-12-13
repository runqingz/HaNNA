import tensorflow as tf
import time

with tf.device('/GPU:0'):

    img = tf.random.uniform([8, 56, 56, 64])
    filters = tf.random.uniform([3, 3, 64, 64])
    depthwise_filter = tf.random.uniform([3, 3, 64, 1])
    pointwise_filter = tf.random.uniform([1, 1, 64, 64])
    
    best = None
    num_trials = 20
    num_iter = 10
    times = []
    for j in range(num_trials):
        start = time.time()
        for i in range(num_iter):
            # 1111
            out = tf.nn.separable_conv2d(
                img, depthwise_filter, pointwise_filter,
                strides = (1, 1, 1, 1), padding = 'SAME')
            
            # 2222
            # out = tf.nn.depthwise_conv2d(img, depthwise_filter, 
            #     strides = (1, 1, 1, 1), padding = 'SAME')
            # out = tf.nn.conv2d(img, filters,
            #     strides = (1, 1, 1, 1), padding = 'SAME')

            # 3333
            # out = tf.nn.conv2d(img, filters, strides = 1, padding='SAME')
        end = time.time()
        t = (end - start) * 1000 / num_iter
        times.append(t)
        if not best or t < best: best = t
    print('time: {} ms'.format(best))
    print('avg time: {} ms'.format(sum(times[2:]) / (len(times) - 2)))