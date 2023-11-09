import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)  # to read only one hand
classifier = Classifier("Model/Keras_model.h5", "Model/labels.txt")   # classifier used to detect the sign that is shown by the user

offset = 25
imgSize = 300

counter = 0

labels = ["0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]   # to show the result according to the index produced by the classifier

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)  # predefined function to read hands

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']   # gives the values of the bounding box of the hand

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255   # creating a neutral image foe the ease of the classifier
        imgCrop = img[y - offset: y + h + offset, x - offset: x + w + offset]  # making hand image wider and taller than the boundary for clarity

        imgCropShape = imgCrop.shape

        aspectRatio = h/w

        if aspectRatio > 1:   # when height is greater than the width
            k = imgSize/h
            wCal = math.ceil(k*w)   # adjust the width according to the height
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize-wCal)/2)
            imgWhite[:, wGap:wCal+wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite)

        else:   # when width is greater than height
            k = imgSize/w
            hCal = math.ceil(k*h)   # adjust the height according to the width
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize-hCal)/2)
            imgWhite[hGap:hCal+hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite)

        cv2.putText(imgOutput, labels[index], (x, y-20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 1)

    cv2.imshow("ImageFinal", imgOutput)
    cv2.waitKey(1)
