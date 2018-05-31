import numpy as np
from PIL import Image
import time
from datetime import datetime
import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import uic

import tensorflow as tf
slim = tf.contrib.slim

form_class = uic.loadUiType("Object_detection_GUI.ui")[0]

mean = [103.939, 116.779, 123.68]

channel_num = 6;
channel_name = ['building', 'road', 'water', 'farm', 'tree', 'other']

def clamp(img):
    img[img<0] = 0
    img[img>255] = 255
    return img

def vgg_arg_scope(weight_decay=0.0005):
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_initializer=tf.zeros_initializer()):
    with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
      return arg_sc

def leaky_relu_001(x):
    return tf.nn.leaky_relu(x, alpha=0.01)

def vgg_16(inputs, is_training=True, keep_prob=0.5, scope='vgg_16'):
  with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d], activation_fn=leaky_relu_001):
      C1 = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      P1 = slim.max_pool2d(C1, [2, 2], scope='pool1')
      C2 = slim.repeat(P1, 2, slim.conv2d, 128, [3, 3], scope='conv2')
      P2 = slim.max_pool2d(C2, [2, 2], scope='pool2')
      C3 = slim.repeat(P2, 3, slim.conv2d, 256, [3, 3], scope='conv3')
      P3 = slim.max_pool2d(C3, [2, 2], scope='pool3')
      C4 = slim.repeat(P3, 3, slim.conv2d, 512, [3, 3], scope='conv4')
      P4 = slim.max_pool2d(C4, [2, 2], scope='pool4')
      C5 = slim.repeat(P4, 3, slim.conv2d, 512, [3, 3], scope='conv5')
      P5 = slim.max_pool2d(C5, [2, 2], scope='pool5')

      C6 = slim.conv2d(P5, 1024, [7, 7], scope='conv6')
      C6D = slim.dropout(C6, keep_prob=keep_prob, is_training=is_training, scope='drop6')
      C7 = slim.conv2d(C6D, 1024, [1, 1], scope='conv7')
      C7D = slim.dropout(C7, keep_prob=keep_prob, is_training=is_training, scope='drop7')

      D5 = slim.conv2d(C7D, 512, [3, 3], scope='deconv5_conv1')
      PC5 = slim.conv2d(P5, 512, [3, 3], scope='deconv5_conv2')
      D4 = slim.layers.conv2d_transpose(tf.concat([D5, PC5], axis=3), 512, [4, 4], [2, 2], padding='SAME', scope='deconv4')
      PC4 = slim.conv2d(P4, 512, [3, 3], scope='deconv4_conv2')
      D3 = slim.layers.conv2d_transpose(tf.concat([D4, PC4], axis=3), 256, [4, 4], [2, 2], padding='SAME', scope='deconv3')
      PC3 = slim.conv2d(P3, 256, [3, 3], scope='deconv3_conv2')
      D2 = slim.layers.conv2d_transpose(tf.concat([D3, PC3], axis=3), 256, [4, 4], [2, 2], padding='SAME', scope='deconv2')
      PC2 = slim.conv2d(P2, 256, [3, 3], scope='deconv2_conv2')
      D1 = slim.layers.conv2d_transpose(tf.concat([D2, PC2], axis=3), 128, [4, 4], [2, 2], padding='SAME', scope='deconv1')
      PC1 = slim.conv2d(P1, 128, [3, 3], scope='deconv1_conv2')
      D0 = slim.layers.conv2d_transpose(tf.concat([D1, PC1], axis=3), 128, [4, 4], [2, 2], padding='SAME', scope='deconv0')
      logits = slim.conv2d(D0, channel_num, [3, 3], scope='logits')

      return logits

def ConvNet(input):
    curr_arg_scope = vgg_arg_scope()
    with slim.arg_scope(curr_arg_scope):
        logits = vgg_16(input, is_training=False, keep_prob=1.0)

    with tf.name_scope('classify'):
        result = tf.nn.softmax(logits, name='result') # softmax applied to last dimension. or, specified by "dim"

    return result

test_fn = ConvNet

def np_to_Q(img):
    temp_PIL = Image.fromarray(np.uint8(img))
    temp_c_show = np.array(temp_PIL.resize((300, 300), Image.ANTIALIAS))
    temp_qim = QImage(temp_c_show.data, temp_c_show.shape[1], temp_c_show.shape[0], temp_c_show.strides[0],QImage.Format_RGB888)
    temp_qpx = QPixmap.fromImage(temp_qim)
    return temp_qpx

class MyWindow(QMainWindow, form_class):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.setupUi(self)

        self.pushButton.clicked.connect(self.btn_clicked_1)
        self.pushButton_2.clicked.connect(self.btn_clicked_2)

        self.input = tf.placeholder(tf.float32, [None, None, None, 3])
        self.prediction = test_fn(self.input)
        self.result = tf.identity(self.prediction, name='result')

        self.sess = tf.InteractiveSession()
        saver = tf.train.Saver(max_to_keep=3)
        saver.restore(self.sess, 'checkpoint/last_checkpoint.ckpt')

        color_b = 255 * np.ones((100, 100, 3)); color_b[:, :, 2] = 0; color_b[:, :, 1] = 0
        color_r = 255 * np.ones((100, 100, 3)); color_r[:, :, 1] = 0; color_r[:, :, 0] = 0
        color_w = 255 * np.ones((100, 100, 3)); color_w[:, :, 1] = 223; color_w[:, :, 0] = 206
        color_f = np.zeros((100, 100, 3)); color_f[:, :, 1] = 255;
        color_t = np.zeros((100, 100, 3)); color_t[:, :, 1] = 80;

        self.label_7.setPixmap(np_to_Q(color_b))
        self.label_9.setPixmap(np_to_Q(color_r))
        self.label_11.setPixmap(np_to_Q(color_w))
        self.label_13.setPixmap(np_to_Q(color_f))
        self.label_15.setPixmap(np_to_Q(color_t))

    def btn_clicked_1(self):
        self.fn = QFileDialog.getOpenFileName(self, 'image', QDir.currentPath())[0]
        temp_sat_PIL = Image.open(str(self.fn))
        temp_sat = np.array(temp_sat_PIL)
        temp_sat_ = np.copy(temp_sat).astype('float32')
        for ch in range(3):
            temp_sat_[..., ch] -= mean[ch]
        th, tw, tc = temp_sat_.shape
        img_input = np.zeros((1, th, tw, 3))
        img_input[0, :, :, :] = temp_sat_
        self.c0 = img_input.astype('float32')
        self.c1 = np.copy(temp_sat)
        self.c1_show = np.array(temp_sat_PIL.resize((300, 300), Image.ANTIALIAS))
        self.qim1 = QImage(self.c1_show.data, self.c1_show.shape[1], self.c1_show.shape[0], self.c1_show.strides[0], QImage.Format_RGB888)
        self.qpx1 = QPixmap.fromImage(self.qim1)
        self.label.setPixmap(self.qpx1)
        self.label_2.clear()

        h, w = self.c1.shape[:2]
        self.label_3.setText(str(h) + 'x' + str(w))

    def btn_clicked_2(self):
        tic = time.clock()

        test_result = self.sess.run(self.result, feed_dict={self.input: self.c0})

        tb, th, tw, tc = test_result.shape
        test_img = np.zeros((th, tw, 3))
        test_img_ = test_result[0, ...]
        test_back = np.zeros((th, tw))
        for j in range(channel_num-1):
            test_back += test_img_[:, :, j]

        test_norm = np.copy(test_back)
        test_norm[test_norm < 1.0] = 1.0

        test_back[test_back > 1.0] = 1.0
        test_back = 1 - test_back

        for j in range(channel_num):
            test_img_[..., j] /= test_norm

        test_img[:, :, 0] = test_img_[:, :, 0] *  0  + test_img_[:, :, 1] * 255 + test_img_[:, :, 2] * 255 + test_img_[:, :, 3] *  0  + test_img_[:, :, 4] *  0 + test_back * 255
        test_img[:, :, 1] = test_img_[:, :, 0] *  0  + test_img_[:, :, 1] *  0  + test_img_[:, :, 2] * 223 + test_img_[:, :, 3] * 255 + test_img_[:, :, 4] * 80 + test_back * 255
        test_img[:, :, 2] = test_img_[:, :, 0] * 255 + test_img_[:, :, 1] *  0  + test_img_[:, :, 2] * 206 + test_img_[:, :, 3] *  0  + test_img_[:, :, 4] *  0 + test_back * 255

        toc = time.clock()
        process_time = '%4.2f(s)' % (toc-tic)

        self.label_2.setPixmap(np_to_Q(test_img[..., ::-1]))
        self.label_5.setText(process_time)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = MyWindow()
    myWindow.show()
    app.exec_()
