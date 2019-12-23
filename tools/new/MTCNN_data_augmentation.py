import cv2
import os
import numpy as np

txt = '/home/hzouaghi/Documents/DATAs/Face_detection/UTK_face/UTKface_inthewild/UTKface_inthewild/landmark_list_part1.txt'
augmented_path = 'landmark_list_part1_augmented.txt'
data_part="UTK_part1/"

from mtcnn.mtcnn import MTCNN
import face_recognition

# initialise the detector class.
detector = MTCNN()

# load an image as an array

dirname = os.path.dirname(txt)


augmented = open(os.path.join(dirname, augmented_path) ,"w+")



#for line in open(txt, 'r'):
for index,line in enumerate(open(txt, 'r')):
    line = line.strip()
    components = line.split(' ')
    img_path = os.path.join(dirname, data_part+components[0])  # file path

    # prepare the image
    image = face_recognition.load_image_file(img_path)
    # detect faces from input image.
    face_locations = face_recognition.face_locations(image,model='hog')#, model="cnn")

    if(face_locations):
        # Detected bounding box
        bbox_str="{} {} {} {}".format(face_locations[0][1],face_locations[0][3],face_locations[0][2],face_locations[0][0])
        cv2.circle(image, (int(face_locations[0][1]),int(face_locations[0][2])), 3, (0,0,0))
        cv2.circle(image, (int(face_locations[0][3]),int(face_locations[0][0])), 3, (0,0,0))
        #print (bbox_str)
        print(line)
        # show landmarks
        lands_2d = np.array(components[1:]).reshape(-1,2)
        for pt in lands_2d:
            cv2.circle(image, (int(pt[0]),int(pt[1])), 3, (0,0,255))

        # load landmarks
        landmarks=" ".join(components[1:])
        #print (landmarks)

        #write the line
        #line = "{} {} {}\n".format(data_part+components[0],bbox_str,landmarks)
        #augmented.write(line)        
        #print(line)

        #show in opencv
        cv2.imshow('image',image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if(index>5):
            break
    else :
        print ("-----------------------------Bouniding box not found")
'''






'''
