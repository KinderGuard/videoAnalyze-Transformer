import TransformerNet
from TransformerNet import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import cv2
import imutils

plt.ion()

def count_frames_manual(video):
    # initialize the total number of frames read
    total = 0
    # loop over the frames of the video
    while True:
        # grab the current frame
        (grabbed, frame) = video.read()

        # check to see if we have reached the end of the
        # video
        if not grabbed:
            break
        # increment the total number of frames read
        total += 1
    # return the total number of frames in the video file
    return total


cap = cv2.VideoCapture('./videos/test_video1.mp4')
total = count_frames_manual(cap)
TransformerNet.Semi_Transformer(2, total)

# params for ShiTomasi corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.1,
                      minDistance=7,
                      blockSize=7)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
#old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_frame, mask=None, **feature_params)  # Shi-Tomasi의 코너점
# print(p0)  # shape : (코너점 갯수, 1, 2)[[[a,b]], ..., [[c,d]]]

while True:
    ret, frame = cap.read()
    #blank_image = None  # 빈 이미지
    # print(frame.shape) # 높이, 너비, 채널의 수
    if ret:
        #frame = imutils.resize(frame, width=800)  # frame 크기 재조정
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, weight, channel = frame.shape
        #blank_image = np.zeros((height, weight, 3), np.uint8)  # frame과 크기가 같은 빈 이미지 만들기

        # param : 이전 프레임, 추적할 이전 포인트, 다음 프레임 등을 인자로 전달
        # return : 이전 프레임에서 추적할 포인트가 연속된 다음 프레임에서 추적될 경우 상태값 1을 반환, 아님 0
        #          추적할 포인트가 이동한 새로운 위치값도 함께 반환
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_frame, frame_gray, p0, None, **lk_params)  # 이전 프레임, 이전 점, 다음 프레임
        # Select good points
        good_new = p1[st == 1]  # 다음 프레임에서 추척할 포인트가 이동한 위치값들의 배열.

        for new in good_new:
            y, f = TransformerNet.Semi_Transformer.forward(new)
            print(y)
            print('\n')