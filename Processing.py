import dlib
import cv2
import os
import re
import json
from pylab import *
from PIL import Image, ImageChops, ImageEnhance

train_frame_folder = 'train_sample_videos'
#with open(os.path.join(train_frame_folder, 'metadata.json'), 'r') as file:
with open(os.path.join(train_frame_folder ,'metadata.json'), 'r') as file:
    data = json.load(file)
    print(data)
list_of_train_data = [f for f in os.listdir(train_frame_folder) if f.endswith('.mp4')]
detector = dlib.get_frontal_face_detector()
for vid in list_of_train_data:
    count = 0
    cap = cv2.VideoCapture(os.path.join(train_frame_folder, vid))
    frameRate = cap.get(5)
    while cap.isOpened():
        frameId = cap.get(1)
        ret, frame = cap.read()
        if ret != True:
            break
        if frameId % ((int(frameRate)+1)*1) == 0:
            face_rects, scores, idx = detector.run(frame, 0)
            for i, d in enumerate(face_rects):
                x1 = d.left()
                y1 = d.top()
                x2 = d.right()
                y2 = d.bottom()
                crop_img = frame[y1:y2, x1:x2]
                if data[vid]['label'] == 'REAL':
                    path = 'datasets/real'
                    cv2.imwrite(os.path.join(path, vid.split('.')[0]+'_'+str(count)+'.png'), cv2.resize(crop_img, (128, 128)))
                    #cv2.imwrite('dataset/real/'+vid.split('.')[0]+'_'+str(count)+'.png', cv2.resize(crop_img, (128, 128)))
                elif data[vid]['label'] == 'FAKE':
                    path = 'datasets/fake'
                    cv2.imwrite(os.path.join(path, vid.split('.')[0]+'_'+str(count)+'.png'), cv2.resize(crop_img, (128, 128)))
                    #cv2.imwrite('dataset/fake/'+vid.split('.')[0]+'_'+str(count)+'.png', cv2.resize(crop_img, (128, 128)))
                count+=1