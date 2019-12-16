import cv2
import os
import numpy as np

txt = '/home/hzouaghi/Documents/deepneuralnetwork/mtcnn_tf/dataset/landmark_list_aumented_part3_1.txt'


from mtcnn.mtcnn import MTCNN
import face_recognition

# initialise the detector class.
detector = MTCNN()

# load an image as an array

dirname = os.path.dirname(txt)


#for line in open(txt, 'r'):
for index,line in enumerate(open(txt, 'r')):
    line = line.strip()
    components = line.split(' ')
    img_path = os.path.join(dirname, components[0])  # file path

    # prepare the image
    image = face_recognition.load_image_file(img_path)
    # detect faces from input image.

    # Detected bounding box
    cv2.circle(image, (int(components[1]),int(components[3])), 3, (0,255,0))
    cv2.circle(image, (int(components[2]),int(components[4])), 3, (0,255,0))

    # Loaded bounding
    cv2.circle(image, (int(components[5]),int(components[7])), 3, (0,0,255))
    cv2.circle(image, (int(components[6]),int(components[8])), 3, (0,0,255))

    #show in opencv
    cv2.imshow('image',image)
    cv2.waitKey(0)


    if(index>5):
        break
cv2.destroyAllWindows()
'''






'''
