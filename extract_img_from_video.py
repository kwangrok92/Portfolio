import os
import cv2

# 저장할 폴더 생성 (있으면 line6 주석처리)
save_dir = './cap_video2'
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
        cv2.imwrite(os.path.join(save_dir, "frame{:d}.jpg".format(count+1)), image)  # save frame as JPEG file
        print('Capturing img... please wait')
    count += 1

    if count >= 12354:
        print('Completed')
