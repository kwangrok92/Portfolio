import os
import cv2

# 저장할 폴더 생성
save_dir = './cap_video'
# os.mkdir(save_dir)

# 이미지 추출 from 동영상
vidcap = cv2.VideoCapture('./target_video/luxury_taeim.mp4')
count = 0

while True:
    success, image = vidcap.read()
    if not success:
        break

    # 이미지 저장
    if count % 10 == 0:
        cv2.imwrite(os.path.join(save_dir, "frame{:d}.jpg".format(count)), image)  # save frame as JPEG file
        print('Write a new frame:', count+1)
    count += 1