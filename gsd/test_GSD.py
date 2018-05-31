import numpy as np
from PIL import Image
import time

import tensorflow as tf
slim = tf.contrib.slim
import mobilenet_v1

fn = '05_sat.png'

def clamp(img):
    img[img<0] = 0
    img[img>255] = 255
    return img

def mobilenet(inputs,
              num_classes=1000,
              dropout_keep_prob=0.999,
              is_training=True,
              min_depth=8,
              depth_multiplier=1.0,
              conv_defs=None,
              spatial_squeeze=True,
              reuse=None,
              scope='MobilenetV1',
              global_pool=False):
  input_shape = inputs.get_shape().as_list()
  if len(input_shape) != 4:
    raise ValueError('Invalid input tensor rank, expected 4, was: %d' %
                     len(input_shape))

  with tf.variable_scope(scope, 'MobilenetV1', [inputs], reuse=reuse) as scope:
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=is_training):
      nets = mobilenet_v1.mobilenet_v1_base(inputs, scope=scope,
                                            min_depth=min_depth,
                                            depth_multiplier=depth_multiplier,
                                            conv_defs=conv_defs)
      net = nets[0]

  return net

def ConvNet(input):
    curr_arg_scope = mobilenet_v1.mobilenet_v1_arg_scope()
    with slim.arg_scope(curr_arg_scope):
        nets = mobilenet(input, is_training=False)

    logits = slim.conv2d(nets, 1, 1, activation_fn=tf.identity, scope='outputR', padding="VALID")

    with tf.name_scope('regress'):
        result = tf.identity(logits, name='result')

    return result

test_fn = ConvNet

input = tf.placeholder(tf.float32, [None, None, None, 3])
prediction = test_fn(input)
result = tf.identity(prediction, name='result')

sess = tf.InteractiveSession()
saver = tf.train.Saver(max_to_keep=3)
saver.restore(sess, 'checkpoint/last_checkpoint.ckpt')

temp_sat_PIL = Image.open(fn)
temp_sat = np.array(temp_sat_PIL)
temp_sat_ = np.copy(temp_sat) / 127.5 - 1.0
th, tw, tc = temp_sat_.shape
img_input = np.zeros((1, th, tw, 3))
img_input[0, :, :, :] = temp_sat_
c0 = img_input.astype('float32')

tic = time.clock()

res_pre_ = sess.run(result, feed_dict={input: c0})
res_pre = np.median(res_pre_[0, :, :, :])

toc = time.clock()
process_time = '%4.2f(s)' % (toc-tic)

temp_sat_PIL.save('test_(%4.2f).png' % res_pre)
