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
import mobilenet_v1

form_class = uic.loadUiType("Pixel_resolution_estimation_GUI.ui")[0]

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

class MyWindow(QMainWindow, form_class): # 63~ 70 부분은 ui 초기화하고 button을 초기화 하는 부분이다. 신경쓸 필요 없음
    def __init__(self):    # 혹시라도 수정한다면 pyqt5 tutorial을 보면 된다.
        super(MyWindow, self).__init__()
        self.setupUi(self)
        #self.connect(self.pushButton, SIGNAL("clicked()"), self.btn_clicked_1)
        #self.connect(self.pushButton_2, SIGNAL("clicked()"), self.btn_clicked_2)
        self.pushButton.clicked.connect(self.btn_clicked_1)
        self.pushButton_2.clicked.connect(self.btn_clicked_2)

        self.input = tf.placeholder(tf.float32, [None, None, None, 3])
        self.prediction = test_fn(self.input)
        self.result = tf.identity(self.prediction, name='result')

        self.sess = tf.InteractiveSession()
        saver = tf.train.Saver(max_to_keep=3)
        saver.restore(self.sess, 'checkpoint/last_checkpoint.ckpt')

    def btn_clicked_1(self): # 버튼 누를 때 창 열고 가져오는 것을 하는 부분이다.
        self.fn = QFileDialog.getOpenFileName(self, 'image', QDir.currentPath())[0]
        temp_sat_PIL = Image.open(str(self.fn))
        temp_sat = np.array(temp_sat_PIL)
        temp_sat_ = np.copy(temp_sat) / 127.5 - 1.0
        th, tw, tc = temp_sat_.shape
        img_input = np.zeros((1, th, tw, 3))
        img_input[0, :, :, :] = temp_sat_
        self.c0 = img_input.astype('float32')
        self.c1 = np.copy(temp_sat)
        self.c1_show = np.array(temp_sat_PIL.resize((300, 300), Image.ANTIALIAS))
        self.qim1 = QImage(self.c1_show.data, self.c1_show.shape[1], self.c1_show.shape[0], self.c1_show.strides[0], QImage.Format_RGB888) # pyqt기준으로 이미지 출력 해주는 부분
        self.qpx1 = QPixmap.fromImage(self.qim1)
        self.label.setPixmap(self.qpx1)
        self.label_3.clear()
        self.label_5.clear()

    def btn_clicked_2(self):
        tic = time.clock()

        res_pre_ = self.sess.run(self.result, feed_dict={self.input: self.c0})
        res_pre = np.median(res_pre_[0, :, :, :])

        toc = time.clock()
        process_time = '%4.2f(s)' % (toc-tic)
        res_pre_text = '%4.2f (cm/pixel)' % res_pre

        self.label_3.setText(res_pre_text)
        self.label_5.setText(process_time)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = MyWindow()
    myWindow.show()
    app.exec_()
