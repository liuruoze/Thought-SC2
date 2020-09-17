import numpy as np
import tensorflow as tf

# usage of ortho_init
#w = tf.get_variable("w", [nin, nh], initializer=ortho_init(init_scale))
#b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(init_bias))


def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        #lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4: # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init


def get_initializer(in_dim, out_dim, initial_type='original'):
    d = 1. / 500
    normal = np.sqrt(1. / in_dim)
    xavier = np.sqrt(6. / (in_dim + out_dim))
    he = np.sqrt(2. / in_dim)

    if initial_type == 'original' :
        w_init = tf.random_uniform_initializer(-d, d)
        b_init = tf.random_uniform_initializer(-d, d)
    elif initial_type == 'normal' :
        w_init = tf.random_normal_initializer(-normal, normal)
        b_init = tf.random_normal_initializer(-normal, normal)        
    elif initial_type == 'xavier':             
        w_init = tf.random_uniform_initializer(-xavier, xavier)
        b_init = tf.random_uniform_initializer(-xavier, xavier)
    elif initial_type == 'he' :
        w_init = tf.random_normal_initializer(stddev=he)
        b_init = tf.zeros_initializer()
    else:
        w_init = tf.zeros_initializer()
        b_init = tf.zeros_initializer()

    return w_init, b_init


def output_layer(data, out_dim, add_dim, name, func=tf.nn.relu, trainable=True, norm=False, is_training=False, initial_type='original'):
    in_dim = data.get_shape().as_list()[-1]
    shape = [in_dim, out_dim]
    shape_add = [in_dim, add_dim]

    with tf.variable_scope(name):
        w_init, b_init = get_initializer(in_dim, add_dim, initial_type=initial_type)

        w = tf.get_variable(name="weights", shape=shape, initializer=w_init, trainable=trainable)
        b = tf.get_variable(name="bias", shape=[out_dim], initializer=b_init, trainable=trainable)

        output = tf.matmul(data, w) + b

        if add_dim > 0:
            with tf.variable_scope("add_output_layer"):
                w_init_add, b_init_add = get_initializer(in_dim, add_dim, initial_type=initial_type)
                w_add = tf.get_variable(name="weights", shape=shape_add, initializer=w_init_add, trainable=trainable)
                b_add = tf.get_variable(name="bias", shape=[add_dim], initializer=b_init_add, trainable=trainable)

                output_add = tf.matmul(data, w_add) + b_add

            output = tf.concat([output, output_add], -1)

        if norm:
            output = batch_norm(output, is_training)

        if func is not None:
            output = func(output)

    return output


def dense_layer(data, out_dim, name, func=tf.nn.relu, trainable=True, norm=False, is_training=False, initial_type='original'):
    in_dim = data.get_shape().as_list()[-1]
    shape = [in_dim, out_dim]

    with tf.variable_scope(name):
        w_init, b_init = get_initializer(in_dim, out_dim, initial_type)

        w = tf.get_variable(name="weights", shape=shape, initializer=w_init, trainable=trainable)
        b = tf.get_variable(name="bias", shape=[out_dim], initializer=b_init, trainable=trainable)

        output = tf.matmul(data, w) + b

        if norm:
            output = batch_norm(output, is_training)

        if func is not None:
            output = func(output)

    return output


def conv2d_layer(data, filter_size, filter_num, name, stride=1, func=tf.nn.relu, trainable=True):
    in_dim = data.get_shape().as_list()[-1]
    shape = [filter_size, filter_size, in_dim, filter_num]
    d = 1.0 / np.sqrt(filter_size * filter_size * in_dim)

    with tf.variable_scope(name):
        w_init = tf.random_uniform_initializer(-d, d)
        b_init = tf.random_uniform_initializer(-d, d)

        w = tf.get_variable(name="weights", shape=shape, initializer=w_init, trainable=trainable)
        b = tf.get_variable(name="bias", shape=[filter_num], initializer=b_init, trainable=trainable)

        output = tf.nn.conv2d(data, w, strides=[1, stride, stride, 1], padding='SAME', data_format="NHWC") + b
        if func is not None:
            output = func(output)

    return output


def max_pool(data, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'):
    val = tf.nn.max_pool(data, ksize=ksize, strides=strides, padding=padding)

    return val


def batch_norm(x, is_training=True, scope='batch_norm'):
    return tf.contrib.layers.batch_norm(x,
                                        decay=0.9, epsilon=1e-05,
                                        center=True, scale=True, updates_collections=None,
                                        is_training=is_training, scope=scope)


def batch_norm_2(data, is_training, scope='batch_norm'):
    data_norm = tf.contrib.layers.batch_norm(data, is_training=is_training, 
                                                 zero_debias_moving_mean=True, decay=0.9, scope=scope)

    return data_norm
