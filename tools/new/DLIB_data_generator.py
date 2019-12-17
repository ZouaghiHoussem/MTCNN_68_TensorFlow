
 
# import the necessary packages
from imutils import face_utils
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb

import numpy as np
import argparse
import imutils #adrian librery 
import dlib
import cv2
import os

'''
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", default="./shPredictor.dat",
    help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
    help="path to input image")
ap.add_argument("-o", "--output", required=True,
    help="name of output image")
args = vars(ap.parse_args())
'''
#parent folder
path = '/home/hzouaghi/Documents/DATAs/Face_detection/300wDlib'
image_source_folder='UTK'
image_save_folder='UTK_cropped_train'
# output file
output_file_path = 'UTK_dlib_train_dataset.txt'
showImage = False


#train folder
source_path= os.path.join(path, image_source_folder)
export_path= os.path.join(path, image_save_folder)

output_file = open(os.path.join(path, output_file_path) ,"w+")

images_paths = [f for f in os.listdir(source_path) if os.path.isfile(os.path.join(source_path, f))]

images_paths = np.array(images_paths)
np.random.shuffle(images_paths)


def crop_align_image(image,predictor, desiredWidth):

    aligner = FaceAligner(predictor, desiredFaceWidth=desiredWidth)
    image = imutils.resize(image, width=800)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 2)
    if(rects):
        rect = rects[0]
        (x, y, w, h) = rect_to_bb(rect)
        try:
            faceOrig = imutils.resize(image[y:y + h, x:x + w], width=desiredWidth)
            faceAligned = aligner.align(image, gray, rect)
            return faceAligned
        except:
            print("An exception occurred")

    
    print ("------------ Faces was not found -----------------") 
    return False


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
shape_predictor_path = os.path.join(os.path.abspath('./'),'shPredictor.dat')
predictor = dlib.shape_predictor(shape_predictor_path)


max_files=12800
for _index,image_path in enumerate(images_paths):
    
    # load the input image, resize it, and convert it to grayscale
    _path=os.path.join(source_path, image_path)

    image = cv2.imread(_path)
    #image = imutils.resize(image, width=250)
    image= crop_align_image(image,predictor,250)
    if( isinstance(image, bool)):
        continue 
    #image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 2)


    # loop over the face detections
    if (rects):
        print (os.path.join(image_save_folder,image_path))

        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        rect=rects[0]
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        bbox = "{} {} {} {}".format(x, x+w, y ,y+h) # bounding box
        

        # and draw them on the image
        landmarks = " ".join(["%s" % number for number in shape.flatten()])
        

        line = "{} {} {}\n".format(os.path.join(image_save_folder,image_path),bbox,landmarks)
        output_file.write(line)

        if(showImage):
            #drow bbox
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            #drow corners
            cv2.circle(image, (x, y), 2, (0,255,0), -1) #top right corner
            cv2.circle(image,  (x + w, y + h), 2, (0,0,255), -1) # bottom left corner

            # loop over the (x, y)-coordinates for the facial landmarks
            for (x, y) in shape:
                cv2.circle(image, (x, y), 1, (255,255,0), -1)
            cv2.imshow("Output", image)
            cv2.waitKey(0)

    else:
        print ("------------ Faces was not found -----------------") 

    cv2.imwrite(os.path.join(export_path,image_path),image)

    if (_index>max_files):
        break

'''



# show the output image with the face detections + facial landmarks
cv2.imshow("Output", image)

#Saving the image 
ImageName = args["output"]
if(not os.path.isfile(ImageName)):
    cv2.imwrite(ImageName,image)
    print ("Image was saved")
else:
    print ("Image exist already!")


cv2.waitKey(0)
'''