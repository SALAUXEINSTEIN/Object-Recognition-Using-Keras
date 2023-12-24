import cv2 as cv
from cvzone.ClassificationModule import Classifier

mydata = Classifier('C:/Users/HP/Desktop/data/keras_model.h5','C:/Users/HP/Desktop/data/labels.txt')
#mydata = Classifier('data/keras_model.h5','data/labels.txt')
cap = cv.VideoCapture(0)



while True:
    _,img = cap.read()

    predict, index = mydata.getPrediction(img,color=(0,0,255))
    print(predict,index)

    cv.imshow('frame',img)
    key = cv.waitKey(1)
    if key == 27:
        break

