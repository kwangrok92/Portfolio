from PIL import Image
import face_recognition
import os

# 저장할 폴더 생성
os.mkdir("./cap_video_elijah1_crop")

read_path = "./cap_video_elijah1/"
write_path = "./cap_video_elijah1_crop/"
n_face = 0

img = os.listdir(read_path)

for i in range(1, len(img)+1):

    # Load the jpg file into a numpy array
    image = face_recognition.load_image_file(os.path.join(read_path, img[i - 1]))

    # Find all the faces in the image using the default HOG-based model.
    face_locations = face_recognition.face_locations(image)

    print("I found {} face(s) in this photograph.".format(len(face_locations)))
    n_face += len(face_locations)

    for j, face_location in enumerate(face_locations):
        # Print the location of each face in this image
        top, right, bottom, left = face_location
        # print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

        # crop the image
        face_image = image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)

        # resize the image
        pil_image_rs = pil_image.resize(size=(64, 64), resample=0, box=None)

        # write the image
        write_dir = write_path
        pil_image_rs.save(write_dir + "elijah_" + str(i) + '_' + str(j + 1) + ".jpg")

print(n_face)
