"""Train a DeepLab v3 plus model using tensorflow."""

import os
import sys
import cv2
import tensorflow as tf
from deeplabv3 import model as deeplab_model
#from tensorflow.python import debug as tf_debug
import utils.utils as Utils
from utils.data_reader import dataLoader
import datetime

class DeepLabV3:

    def __init__(self):
        self.crop_height=512
        self.crop_width=512
        self.channels=3
        self.output_stride=16   #output stride used in the resnet model
        self.weight_decay=1e-4
        self.ignore_label=255
        self.classes=19
        self.initial_lr=7e-3
        self.decay_steps=50000
        self.end_lr=1e-5
        self.power=0.9
        self.batch_size=8
        self.epochs=42
        self.save_summary_steps=500
        self.print_steps=10
        self.save_checkpoint_steps=1500

        self.dataset='/home/xushaohui/workspace_xsh/mydata/CelebA/WaveData105/CelebAMask-HQ'
        self.pre_trained_model='resnet_v2_101/resnet_v2_101.ckpt'  # well-trained resnet101
        self.train_out='./output'
        self.checkpoints=os.path.join(self.train_out,'checkpoints')
        self.samples=os.path.join(self.train_out,'samples')
        self.saved_summary_train_path=os.path.join(self.train_out,'summary')
        self._check()

    def _check(self):
        os.makedirs(self.checkpoints,exist_ok=True)
        os.makedirs(self.samples,exist_ok=True)
        os.makedirs(self.saved_summary_train_path,exist_ok=True)

    def cal_loss(self,logits, y, loss_weight=1.0):
        y = tf.reshape(y, shape=[-1])
        not_ignore_mask = tf.to_float(tf.not_equal(y, self.ignore_label)) * loss_weight
        one_hot_labels = tf.one_hot(
            y, self.classes, on_value=1.0, off_value=0.0)
        logits = tf.reshape(logits, shape=[-1, self.classes])
        loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=logits, weights=not_ignore_mask)
        return tf.reduce_mean(loss)
        
    def build_model(self):
        with tf.name_scope('input'):
            self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.crop_height, self.crop_width, self.channels], name='x_input')
            self.y = tf.placeholder(dtype=tf.int32, shape=[None, self.crop_height, self.crop_width], name='ground_truth')

        self.logits=deeplab_model.deeplab_v3_plus(self.x,is_training=True,output_stride=self.output_stride, pre_trained_model=self.pre_trained_model)

        with tf.name_scope('regularization'):
            self.train_var_list = [v for v in tf.trainable_variables()
                            if 'beta' not in v.name and 'gamma' not in v.name]
            # Add weight decay to the loss.
            with tf.variable_scope("total_loss"):
                self.l2_loss = self.weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in self.train_var_list])

        with tf.name_scope('loss'):
            self.loss = self.cal_loss(self.logits, self.y)
            tf.summary.scalar('loss', self.loss)
            self.loss_all = self.loss + self.l2_loss
            tf.summary.scalar('loss_all', self.loss_all)

        with tf.name_scope('learning_rate'):
            self.global_step = tf.Variable(0, trainable=False)
            self.lr = tf.train.polynomial_decay(
                learning_rate=self.initial_lr,
                global_step=self.global_step,
                decay_steps=self.decay_steps,
                end_learning_rate=self.end_lr,
                power=self.power,
                cycle=False,
                name=None
            )
            tf.summary.scalar('learning_rate', self.lr)

        with tf.name_scope("opt"):
            self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(self.update_ops):
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9).minimize(self.loss_all, var_list=self.train_var_list, global_step=self.global_step)

        with tf.name_scope("mIoU"):
            self.predictions = tf.argmax(self.logits, axis=-1, name='predictions')
            self.train_mIoU = tf.Variable(0, dtype=tf.float32, trainable=False)
            tf.summary.scalar('train_mIoU', self.train_mIoU)
            self.test_mIoU = tf.Variable(0, dtype=tf.float32, trainable=False)
            tf.summary.scalar('test_mIoU',self.test_mIoU)

        self.merged = tf.summary.merge_all()
    
    def train(self):
        self.build_model()
        with tf.Session() as sess:
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep = 5)
            ckpt = tf.train.get_checkpoint_state(self.checkpoints)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                sess.run(tf.assign(self.global_step, 0))
                print("Model restored...")
            train_summary_writer = tf.summary.FileWriter(self.saved_summary_train_path, sess.graph)
            #test_summary_writer = tf.summary.FileWriter(self.saved_summary_test_path, sess.graph)
            for epoch in range(self.epochs):
                #instance data
                dataPool=dataLoader(dataset=self.dataset,batch_size=self.batch_size,is_training=True)
                idx=0
                for image_batch,anno_batch,batch_filenames in dataPool.get_next():
                    
                    idx+=1
                    logit = sess.run(self.logits, feed_dict={self.x: image_batch, self.y: anno_batch})  #开始训练
                    _ = sess.run(self.optimizer, feed_dict={self.x: image_batch, self.y: anno_batch})  #开始训练
                    if idx % self.save_summary_steps == 0:
                        train_summary = sess.run(self.merged, feed_dict={self.x: image_batch, self.y: anno_batch})
                        train_summary_writer.add_summary(train_summary, idx)
                        #test_summary = sess.run(self.merged, feed_dict={x: image_batch_val, y: anno_batch_val})
                        #test_summary_writer.add_summary(test_summary, i)
                    if idx % self.print_steps == 0:
                        train_loss_val_all = sess.run(self.loss_all, feed_dict={self.x: image_batch, self.y: anno_batch})
                        print(datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S"), " | Step: %d, | Train loss all: %f" % (idx, train_loss_val_all))
                    if idx % self.save_summary_steps == 0:
                        learning_rate = sess.run(self.lr)
                        pred_train, train_loss_val_all, train_loss_val = sess.run([self.predictions, self.loss_all, self.loss],
                                                                                feed_dict={self.x: image_batch, self.y: anno_batch})
                        train_mIoU_val, train_IoU_val = Utils.cal_batch_mIoU(pred_train, anno_batch, self.classes)
                        #test_mIoU_val, test_IoU_val = Utils.cal_batch_mIoU(pred_test, anno_batch_val, self.classes)
                        sess.run(tf.assign(self.train_mIoU, train_mIoU_val))
                        #sess.run(tf.assign(test_mIoU, test_mIoU_val))
                        print('------------------------------')
                        # print(datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S"), " | Step: %d, | Lr: %f, | train loss all: %f, | train loss: %f, | train mIoU: %f, | test loss all: %f, | test loss: %f, | test mIoU: %f" % (
                        #     i, learning_rate, train_loss_val_all, train_loss_val, train_mIoU_val, test_loss_val_all, test_loss_val, test_mIoU_val))
                        print(datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S"), " | Step: %d, | Lr: %f, | train loss all: %f, | train loss: %f, | train mIoU: %f" % (idx, learning_rate, train_loss_val_all, train_loss_val, train_mIoU_val))
                        print('------------------------------')
                        print(train_IoU_val)
                        print('------------------------------')

                    if idx % self.save_checkpoint_steps == 0:
                        os.makedirs(self.checkpoints,exist_ok=True)
                        os.makedirs(self.samples,exist_ok=True)
                        for j in range(self.batch_size):
                            cv2.imwrite('images/%d_%s_train_img.png' %(idx, batch_filenames[j].split('.')[0]), image_batch[j])
                            cv2.imwrite('images/%d_%s_train_anno.png' %(idx, batch_filenames[j].split('.')[0]), Utils.color_gray(anno_batch[j]))
                            cv2.imwrite('images/%d_%s_train_pred.png' %(idx, batch_filenames[j].split('.')[0]), Utils.color_gray(pred_train[j]))
                        saver.save(sess, os.path.join(self.checkpoints, 'deeplabv3plus.model'), global_step=idx*(epoch+1))

if __name__=='__main__':
    deeplab=DeepLabV3()
    deeplab.train()


   