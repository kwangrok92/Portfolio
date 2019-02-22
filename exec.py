from __future__ import print_function

import argparse
import cv2
import face_recognition
import imutils

from ben_train import Model

font = cv2.FONT_HERSHEY_SIMPLEX

model = Model()
model.load()

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-e", "--encodings", default = 'encodings.pickle',
    help = "path to serialized db of facial encodings")

ap.add_argument("-d", "--detection-method", type=str, default="cnn",
    help = "face detection model to use: either `hog` or `hog`")

args = vars(ap.parse_args())

IMAGE_FILE_PATH = './test_data/test_ben_1.jpg'

# 테스트할 이미지 읽어오기
b_im = cv2.imread(IMAGE_FILE_PATH, cv2.IMREAD_ANYCOLOR)

width = b_im.shape[1]  # 너비 가져오기
height = b_im.shape[0]  # 높이 가져오기

# 흑백으로 변경
grayframe = cv2.cvtColor(b_im, cv2.COLOR_BGR2GRAY)

# RGB로 변경
rgb_image = cv2.cvtColor(b_im, cv2.COLOR_BGR2RGB)

# 입력이미지를 BGR에서 RGB로 변경 -> 그리고 Resizing
# a width of 750px (빠른 처리를 위해)
rgb = cv2.cvtColor(b_im, cv2.COLOR_BGR2RGB)
rgb = imutils.resize(b_im, width=420)
r = b_im.shape[1] / float(rgb.shape[1])

# 박스 칠 좌표 찾기
# corresponding to each face in the input frame, then compute
# the facial embeddings for each face
boxes = face_recognition.face_locations(rgb, model = args["detection_method"])

for (top , right , bottom, left) in boxes:
    # rescale the face coordinates
    rect = [top , right , bottom, left]
    top = int(top * r)
    right = int(right * r)
    bottom = int(bottom * r)
    left = int(left * r)
    y = top -2 if top - 15 > 15 else top + 15

image = b_im[top: bottom, left: right]
result, ben_ac = model.predict(image)

if result == 0:  # boss
    print('ben')
    # cv2.rectangle(frame,( left-right//6,top-bottom//6 ),(right+right//6,bottom+bottom//6),(0,255,0),2)
    # cv2.putText(frame, 'Oracle YU : '+str(round(oy,4)), (left-5, top-5), font, 0.8, (153,102,0),2)
    cv2.rectangle(b_im, (left, top), (right, bottom), (100, 255, 255), 2)
    cv2.rectangle(b_im, (left, bottom + 25), (right, bottom), (100, 255, 255), cv2.FILLED)
    cv2.putText(b_im, 'Ben ' + str(round(ben_ac, 4)), (left + 6, bottom+13), font, 0.5, (0, 0, 0), 1)
else:
    print('no')
    # cv2.rectangle(frame,( left-right//6,top-bottom//6 ),(right+right//6,bottom+bottom//6),(0,0,255),2)
    cv2.rectangle(b_im, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.rectangle(b_im, (left, bottom + 25), (right, bottom), (0, 255, 0), cv2.FILLED)
    cv2.putText(b_im, 'No ' + str(round(ben_ac, 4)), (left + 6, bottom +13), font, 0.5, (0, 0, 0), 1)
    
cv2.imwrite('./test_data/test_result.jpg', b_im)
