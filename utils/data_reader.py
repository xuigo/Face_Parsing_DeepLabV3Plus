import numpy as np
import os
from random import shuffle
import cv2
import random 
from random import choice
import math


class dataLoader:

    def __init__(self,dataset,batch_size,is_training=True):
        self.height=512
        self.width =512
        #self.channels = 3
        #self.classes = 21
        #self.min_scale = 0.5
        #self.max_scale = 2.0
        self.scale = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
        self.ignore_label = 255
        self.r_mean = 123.15
        self.g_mean = 115.90
        self.b_mean = 103.06
        #self.mean_rgb = [self.r_mean, self.g_mean, self.b_mean]
        self.dataset=dataset
        self.batch_size=batch_size
        self.image_dir=os.path.join(self.dataset,'CelebA-HQ-img')
        self.anno_dir=os.path.join(self.dataset,'CelebAMask-Manual')
        self._check()   
        self._get_file()
        #self._epochs_done = 0
        #self._index_in_epoch = 0
        #self._flag = 0
        
        self.is_training=is_training

    def _check(self):
        if not os.path.exists(self.image_dir): raise ValueError(" Location:%s doesn't exists"%self.image_dir)
        if not os.path.exists(self.anno_dir): raise ValueError(" Location:%s doesn't exists"%self.anno_dir)

    def _get_file(self):
        #file structure:
        #dataset:
        #|---images:
        #|    |---00000.jpg
        #|---annos:
        #|    |---00000.png
        self.files=[]
        for file in os.listdir(self.image_dir):
            src_full_path=os.path.join(self.image_dir,file)
            filepath,filename=os.path.split(src_full_path)
            anno_full_path=os.path.join(self.anno_dir,'{}.png'.format(os.path.splitext(filename)[0]))
            if not os.path.exists(anno_full_path): assert ValueError("Location:{} doesn't exists!".format(anno_full_path))
            self.files.append((src_full_path,anno_full_path))
        shuffle(self.files)
        

        self.train_num=int(len(self.files)//self.batch_size * 1.0 * self.batch_size)
        #self.val_num=len(self.files)-self.train_num
        self.train_files=self.files[:self.train_num]
        #self.val_files=self.files[self.train_num:]

    @staticmethod
    def _flip_random_left_right(image,anno):
        flag=random.randint(0,1)  # 0 or 1
        if flag:
            return cv2.flip(image,1),cv2.flip(anno,1)
        return image,anno


    def _random_pad_crop(self,image,anno):
        image = image.astype(np.float32)
        height, width = anno.shape
        padded_image_r = np.pad(image[:, :, 0], ((0, np.maximum(height, self.height) - height), (0, np.maximum(width, self.width) - width)), mode='constant', constant_values=self.r_mean)
        padded_image_g = np.pad(image[:, :, 1], ((0, np.maximum(height, self.height) - height), (0, np.maximum(width, self.width) - width)), mode='constant', constant_values=self.g_mean)
        padded_image_b = np.pad(image[:, :, 2], ((0, np.maximum(height, self.height) - height), (0, np.maximum(width, self.width) - width)), mode='constant', constant_values=self.b_mean)
        padded_image = np.zeros(shape=[np.maximum(height, self.height), np.maximum(width, self.width), 3], dtype=np.float32)
        padded_image[:, :, 0] = padded_image_r
        padded_image[:, :, 1] = padded_image_g
        padded_image[:, :, 2] = padded_image_b
        padded_anno = np.pad(anno, ((0, np.maximum(height, self.height) - height), (0, np.maximum(width, self.width) - width)), mode='constant', constant_values=self.ignore_label)
        y = random.randint(0, np.maximum(height, self.height) - self.height)
        x = random.randint(0, np.maximum(width, self.width) - self.width)
        cropped_image = padded_image[y:y+self.height, x:x+self.width, :]
        cropped_anno = padded_anno[y:y+self.height, x:x+self.width]
        return cropped_image, cropped_anno

    def _random_resize(self,image,anno):
        height,width=anno.shape
        scale=choice(self.scale)
        scale_image = cv2.resize(image, (int(scale * width), int(scale * height)), interpolation=cv2.INTER_LINEAR)
        scale_anno = cv2.resize(anno, (int(scale * width), int(scale * height)), interpolation=cv2.INTER_NEAREST)
        return scale_image, scale_anno

    def _mean_substraction(self,image):
        substraction_mean_image = np.zeros_like(image, dtype=np.float32)
        substraction_mean_image[:, :, 0] = image[:, :, 0] - self.r_mean
        substraction_mean_image[:, :, 1] = image[:, :, 1] - self.g_mean
        substraction_mean_image[:, :, 2] = image[:, :, 2] - self.b_mean
        return substraction_mean_image

    def augment(self,img,anno):
        scale_img, scale_anno = self._random_resize(img, anno)
        img = img.astype(np.float32)
        cropped_image, cropped_anno = self._random_pad_crop(scale_img, scale_anno)
        flipped_img, flipped_anno = dataLoader._flip_random_left_right(cropped_image, cropped_anno)
        substracted_img = self._mean_substraction(flipped_img)
        return substracted_img, flipped_anno
    
    def get_next(self):
        for i in range(0,self.train_num,self.batch_size):
            batch_imgs,batch_annos,filenames=[],[],[]
            for ii in range(i,i+self.batch_size):
                img=cv2.imread(self.train_files[ii][0])
                img=img[:,:,::-1]
                #anno = cv2.imread(self.train_files[ii][1], cv2.IMREAD_GRAYSCALE)
                anno = cv2.imread(self.train_files[ii][1])[:,:,0]
                if self.is_training:
                    img, anno = self.augment(img, anno)
                else:
                    img=self._mean_substraction(img)
                batch_imgs.append(img)
                batch_annos.append(anno)
                filenames.append(os.path.basename(self.train_files[ii][0]))
            yield np.array(batch_imgs),np.array(batch_annos),filenames












