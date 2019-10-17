import cv2
import sys
import numpy as np
import datetime
import os
import glob
from retinaface import RetinaFace

detector = RetinaFace(gpu=0)

img_path = 'data/retinaface/val/images'

dir = os.listdir(img_path)
for im in dir:
    img = cv2.imread(os.path.join(img_path, im))

    faces, landmarks = detector.detect(img, scales_index=1, do_flip=True)

    if faces is not None:
        print('find', faces.shape[0], 'faces')
        for i in range(faces.shape[0]):
            box = faces[i].astype(np.int)
            color = (0, 0, 255)
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
            title = "%.2f" % (faces[i][4])
            p3 = (max(box[0], 15), max(box[1], 15))
            cv2.putText(img, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
            if landmarks is not None:
                landmark5 = landmarks[i].astype(np.int)
                for l in range(landmark5.shape[0]):
                    color = (0, 0, 255)
                    if l == 0 or l == 3:
                        color = (0, 255, 0)
                    cv2.circle(img, (landmark5[l][0], landmark5[l][1]), 1, color, 2)
        cv2.imwrite(im, img)
    cv2.imshow('0', img)
    cv2.waitKey(1)
