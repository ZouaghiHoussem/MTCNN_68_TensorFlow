import cv2
import os
import numpy as np

#txt = 'dataset/landmark_list_part2_augmented.txt'#testImageList.txt'

#txt = '/home/hzouaghi/Documents/deepneuralnetwork/mtcnn_tf/dataset/trainImageList.txt'

txt = '/home/hzouaghi/Documents/deepneuralnetwork/MTCNN_68_TensorFlow/dataset/lfw_68_train_dataset.txt'

# load an image as an array

dirname = os.path.dirname(txt)
print(dirname)

#for line in open(txt, 'r'):
for index,line in enumerate(open(txt, 'r')):
    line = line.strip()
    components = line.split(' ')
    img_path = os.path.join(dirname, components[0])  # file path

    # prepare the image
    image = cv2.imread(img_path)

    # detect faces from input image.

    # Detected bounding box
    left= int(components[1])
    top= int(components[2])
    right= int(components[3])
    bottom= int(components[4])

    #show corners
    #cv2.circle(image,(left,right), radius=2, color=(0,255,0))
    #cv2.circle(image,(top,bottom), radius=2, color=(0,0,255))
    
    # show bounding box
    cv2.rectangle(image, (left, right), (top, bottom), (0, 255, 0), 2)

 
    # drow landmarks
    landmarks= np.array(components[5:]).reshape(-1,2)
    for pt in landmarks:
        # create and draw dot
        pt[0] = int(round(float(pt[0])))
        pt[1] = int(round(float(pt[1])))

        dot = cv2.circle(image,(int(pt[0]),int(pt[1])), radius=2, color=(255,0,0))
 
    #show in opencv
    cv2.imshow('image',image)
    cv2.waitKey(0)


    if(index>50):
        break
cv2.destroyAllWindows()
'''






'''
