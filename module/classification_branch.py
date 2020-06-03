import tensorflow as tf
from tensorflow.contrib.layers import batch_norm, flatten
from tflearn.layers.conv import avg_pool_2d
from tensorflow.contrib import slim
import numpy as np
import config
import os


class Recognition(object):
    def __init__(self, rnn_hidden_num=256, keepProb=0.8, weight_decay=1e-5, is_training=True):
        self.rnn_hidden_num = rnn_hidden_num
        self.batch_norm_params = {'decay': 0.997, 'epsilon': 1e-5, 'scale': True, 'is_training': is_training}
        self.keepProb = keepProb if is_training else 1.0
        self.weight_decay = weight_decay
        # self.num_classes = config.NUM_CLASSES
        self.is_training = is_training
        self.drop_rate = 0.5

    def cnn(self, rois):
        with tf.variable_scope('recog/cnn'):
            with slim.arg_scope([slim.conv2d],
                                activation_fn=tf.nn.relu,
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=self.batch_norm_params,
                                weights_regularizer=slim.l2_regularizer(self.weight_decay)):
                conv1 = slim.conv2d(rois, 64, 3, stride=1, padding='SAME')
                conv1 = slim.conv2d(conv1, 64, 3, stride=1, padding='SAME')
                pool1 = slim.max_pool2d(conv1, kernel_size=[2, 2], stride=[2, 2], padding='SAME')
                conv2 = slim.conv2d(pool1, 128, 3, stride=1, padding='SAME')
                conv2 = slim.conv2d(conv2, 128, 3, stride=1, padding='SAME')
                pool2 = slim.max_pool2d(conv2, kernel_size=[2, 2], stride=[2, 2], padding='SAME')
                conv3 = slim.conv2d(pool2, 256, 3, stride=1, padding='SAME')
                conv3 = slim.conv2d(conv3, 256, 3, stride=1, padding='SAME')
                pool3 = slim.max_pool2d(conv3, kernel_size=[2, 2], stride=[2, 2], padding='SAME')

                return pool3

    def Fully_connected(self, rois, units, layer_name):
        with tf.variable_scope('recog/part_{}_fc'.format(layer_name)):
            # aver_pooling_logits = avg_pool_2d(rois[:, :, :, :], (4, 4))
            # print("average_pool:", aver_pooling_logits)
            flat_logits = flatten(rois)
            print("dense:", flat_logits)
            fc_1_logits = tf.layers.dense(inputs=flat_logits, use_bias=True, units=1024)
            print("fc1:", fc_1_logits)
            drop_logits = tf.layers.dropout(inputs=fc_1_logits, rate=self.drop_rate, training=self.is_training)
            fc_2_logits = tf.layers.dense(inputs=drop_logits, use_bias=True, units=units)

            return fc_2_logits

    def Part_extract(self, input_x, layer_name):
        with tf.name_scope(layer_name):
            output_height = int(np.shape(input_x)[1])
            output_weight = int(np.shape(input_x)[2])
            stripe_h = int(output_height / 4)
            logits_list = []

            for i in range(4):  # 4 parts
                feature_part = input_x[:, i * stripe_h: (i + 1) * stripe_h, :, :]
                fc_logit = self.Fully_connected(feature_part, units=230, layer_name=i)
                logits_list.append(fc_logit)
            # no slice
            # feature_part = input_x[:, :, :, :]
            # fc_logit = self.Fully_connected(feature_part, units=230, layer_name='all')
            # logits_list.append(fc_logit)

            print("part level :", logits_list)

            return logits_list

    def build_graph(self, rois, seq_len, class_num):
        # part level
        pred_list = []
        logits = self.Part_extract(rois, layer_name='part')
        pred_list.extend(logits)
        print("output shape: ", pred_list)

        # cnn_feature = self.cnn(rois)
        # print("cnn feature shape" , cnn_feature)
        # # add fully connect layers
        # logits = self.Fully_connected(cnn_feature, class_num)
        # print("output shape: ", logits)

        return pred_list

    def loss(self, label, logits_list):
        # Loss and cost calculation
        loss_list = []
        accuracy_list = []
        for logits in logits_list:
            recognition_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))
            loss_list.append(recognition_loss)
            correct_prediction_p = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
            accuracy_p = tf.reduce_mean(tf.cast(correct_prediction_p, tf.float32))
            accuracy_list.append(accuracy_p)

        recognition_loss = tf.reduce_sum(loss_list)
        accuracy_p = tf.reduce_mean(tf.cast(accuracy_list, tf.float32))

        # recognition_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits_list))
        # correct_prediction_p = tf.equal(tf.argmax(logits_list, 1), tf.argmax(label, 1))
        # accuracy_p = tf.reduce_mean(tf.cast(correct_prediction_p, tf.float32))

        return recognition_loss, accuracy_p
