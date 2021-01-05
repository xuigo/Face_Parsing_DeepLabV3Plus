"""Inference a DeepLab v3 plus model using tensorflow."""

import os
import sys
import cv2
import tqdm
import numpy as np
import tensorflow as tf


from deeplabv3 import model as deeplab_model

SRC=''
DST=''
SELECT_PARTIAL=['skin', 'left_brow', 'right_brow', 'left_eye', 'right_eye', 'eye_glass', 'left_ear','right_ear','nose', 'mouth', 'up_lip', 'low_lip', 'hair']

class DeepLabV3:

    def __init__(self):
        self.crop_height = 512
        self.crop_width = 512
        self.channels = 3
        self.output_stride = 16  # output stride used in the resnet model
        self.r_mean = 123.15
        self.g_mean = 115.90
        self.b_mean = 103.06

        self.category_id = {'background': 0, 'skin': 1, 'left_brow': 2, 'right_brow': 3, 'left_eye': 4, 'right_eye': 5, 'eye_glass': 6, 'left_ear': 7,
                            'right_ear': 8, 'earring': 9, 'nose': 10, 'mouth': 11, 'up_lip': 12, 'low_lip': 13, 'neck': 14, 'necklace': 15, 'cloth': 16, 'hair': 17, 'hat': 18}

        self.label_contours = [(0, 0, 0),  # 0=background
                               # 1=skin, 2=l_brow, 3=r_brow, 4=l_eye, 5=r_eye
                               (128, 0, 0), (0, 128, 0), (128, 128,
                                                          0), (0, 0, 128), (128, 0, 128),
                               # 6=eye_g, 7=l_ear, 8=r_ear, 9=ear_r, 10=nose
                               (0, 128, 128), (128, 128, 128), (64,
                                                                0, 0), (192, 0, 0), (64, 128, 0),
                               # 11=mouth, 12=u_lip, 13=l_lip, 14=neck, 15=neck_l
                               (192, 128, 0), (64, 0, 128), (192, 0,
                                                             128), (64, 128, 128), (192, 128, 128),
                               # 16=cloth, 17=hair, 18=hat
                               (0, 64, 0), (128, 64, 0), (0, 192, 0)]
        
        self.select_partial=SELECT_PARTIAL
        self.select_id=[self.category_id[key] for key in self.select_partial]
        self.pre_trained_model = 'resnet_v2_101/resnet_v2_101.ckpt'  # well-trained resnet101
        self.checkpoints = os.path.join('deeplabv3', 'checkpoints')
        self.sess = tf.Session()
        self.build_model()

    def build_model(self):
        with tf.name_scope('input'):
            self.x = tf.placeholder(dtype=tf.float32, shape=[
                                    None, self.crop_height, self.crop_width, self.channels], name='x_input')
        self.logits = deeplab_model.deeplab_v3_plus(
            self.x, is_training=True, output_stride=self.output_stride, pre_trained_model=self.pre_trained_model)

        with tf.name_scope("mIoU"):
            self.predictions = tf.argmax(
                self.logits, axis=-1, name='predictions')

        self.sess.run(tf.local_variables_initializer())
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(self.checkpoints)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("Model restored...")

    def _mean_substraction(self, image):
        substraction_mean_image = np.zeros_like(image, dtype=np.float32)
        substraction_mean_image[:, :, 0] = image[:, :, 0] - self.r_mean
        substraction_mean_image[:, :, 1] = image[:, :, 1] - self.g_mean
        substraction_mean_image[:, :, 2] = image[:, :, 2] - self.b_mean
        return substraction_mean_image

    def load_image(self, path):
        src = cv2.resize(cv2.imread(path), (512, 512))
        src = src[:, :, ::-1]
        img = self._mean_substraction(src)
        return src[np.newaxis, :],img[np.newaxis, :]

    def backProcess(self, image,src,mask_save):
        image=np.squeeze(image)[:,:,::-1]
        dst_mask = np.zeros(
            shape=[self.crop_height, self.crop_width, self.channels])
        dst_res = np.zeros(
            shape=[self.crop_height, self.crop_width, self.channels])
        src = np.squeeze(src)
        height, width = src.shape
        for i in range(height):
            for j in range(width):
                if mask_save: dst_mask[i][j] = self.label_contours[src[i][j]]
                if src[i][j] in self.select_id:  
                    dst_res[i][j] = image[i][j]
        return dst_mask,dst_res

    @staticmethod
    def _get_files(srcPath):
        files = []
        if os.path.isdir(srcPath):
            return [os.path.join(srcPath, file) for file in os.listdir(srcPath)]
        elif os.path.isfile(srcPath):
            return [srcPath]
        else:
            raise ValueError('[!] Unavailable File Location: %s!' % srcPath)

    def inference(self, src, dst,mask_save=True):
        files = DeepLabV3._get_files(src)
        os.makedirs(dst, exist_ok=True)
        for file in tqdm.tqdm(files):
            filepath, filename = os.path.split(file)
            src,image = self.load_image(file)
            preds = self.sess.run(self.predictions, feed_dict={self.x: image})
            mask,result = self.backProcess(src,preds,mask_save)
            if mask_save: cv2.imwrite(os.path.join(dst,'MASK_'+filename), mask)
            cv2.imwrite(os.path.join(dst,filename), result)


if __name__ == '__main__':
    deeplab = DeepLabV3()
    deeplab.inference(SRC,DST)
