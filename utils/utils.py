import cv2
import numpy as np
import sys
sys.path.append("..")

CLASS_NAMES = ['background', 'skin', 'left_brow', 'right_brow', 'left_eye', 'right_eye', 'left_ear', 'right_ear','nose', 'mouth', 'up_lip', 'low_lip', 'hair','eye_glass']
LABEL_CONTOURS = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128),
                               (128, 0, 128), (0, 128, 128), (128,128, 128), (64, 0, 0), (192, 0, 0),
                               (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0)]

def cal_batch_mIoU(pred, gt, classes_num):
    IoU_0 = []
    IoU = []
    eps = 1e-6

    pred_flatten = np.reshape(pred, -1)
    gt_flatten = np.reshape(gt, -1)

    #print(pred_flatten.shape, gt_flatten.shape)

    for i in range(0, classes_num):
        a = [pred_flatten == i, gt_flatten != 255]
        a = np.sum(np.all(a, 0))
        b = np.sum(gt_flatten == i)
        c = [pred_flatten == i, gt_flatten == i]
        c = np.sum(np.all(c, 0))
        iou = c / (a + b - c + eps)
        if b != 0:
            IoU.append(iou)
        IoU_0.append(round(iou, 2))

    IoU_0 = dict(zip(CLASS_NAMES[0:], IoU_0))
    mIoU = np.mean(IoU)
    return mIoU, IoU_0



def color_gray(image):
    cmap = LABEL_CONTOURS
    height, width = image.shape

    return_img = np.zeros([height, width, 3], np.uint8)
    for i in range(height):
        for j in range(width):
            if image[i, j] == 255:
                return_img[i, j, :] = (255, 255, 255)
            else:
                return_img[i, j, :] = cmap[image[i, j]]

    return return_img

if __name__ == '__main__':
    pass