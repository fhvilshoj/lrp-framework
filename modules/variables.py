'''
@author: Vignesh Srinivasan
@author: Sebastian Lapuschkin
@author: Gregoire Montavon
@maintainer: Vignesh Srinivasan
@maintainer: Sebastian Lapuschkin
@contact: vignesh.srinivasan@hhi.fraunhofer.de
@date: 20.12.2016
@version: 1.0+
@copyright: Copyright (c) 2016-2017, Vignesh Srinivasan, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
@license : BSD-2-Clause
'''

import tensorflow as tf




def weights(weights_shape, initializer=tf.truncated_normal_initializer(stddev=0.01), name=''):
    weights_shape = weights_shape
    weights_name = name + '/weights'
    print weights_name
    return tf.get_variable(weights_name, shape=weights_shape, initializer=initializer)


def biases( bias_shape, initializer = tf.constant_initializer(0.0), name = ''):
    bias_shape = bias_shape
    biases_name = name + '/biases'
    print biases_name
    return tf.get_variable(biases_name, bias_shape, initializer=initializer)

