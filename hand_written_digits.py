#!/usr/bin/env python3
import numpy as np
from numpy.core.fromnumeric import resize
import tensorflow as tf
import cv2


# load your model here MNIST-CNN.model
model = tf.keras.models.load_model(
    "MNIST-CNN.model")

# # save your result video in .avi form
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output2.avi', fourcc, 20.0, (640, 480))

#
# Import only if not previously imported
# In VideoCapture object either Pass address of your Video file
# Or If the input is the camera, pass 0 instead of the video file


def getContours(binary_image):
    contours, hierarchy = cv2.findContours(binary_image,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
    return contours


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def process_contours(binary_image, rgb_image, contours):
    black_image = binary_image
    cv2.drawContours(rgb_image, contours, -1, (255, 0, 0), 1)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(rgb_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        leng = int(h * 1.6)
        pt1 = abs(int(y + h // 2 - leng // 2))
        pt2 = abs(int(x + w // 2 - leng // 2))
        roi = black_image[pt1:pt1+leng, pt2:pt2+leng]
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        cv2.imshow("roi", roi)
        roi = roi.reshape(1, 28, 28, 1)
        roi = np.array(roi, dtype='float32')
        roi /= 255
        pred_array = model.predict(roi)
        pred_array = np.argmax(pred_array)
        print('Result: {0}'.format(pred_array))
        cv2.putText(rgb_image, str(pred_array),
                    (x, y), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 3)

    cv2.imshow("Black Image Contours", rgb_image)


cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

if cap.isOpened() == False:
    print("Error in opening video stream or file")
while(cap.isOpened()):
    ret, frame = cap.read()
    # frame = cv2.resize(resize, (640, 480))
    if ret:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        yellowLower = (0, 191, 200)
        yellowUpper = (179, 255, 255)
        mask = cv2.inRange(hsv, yellowLower, yellowUpper)
        # Display the resulting frame
        # cv2.imshow('Frame', mask)

        contours = getContours(mask)
        process_contours(mask, frame, contours)

        # Press esc to exit
        if cv2.waitKey(20) & 0xFF == 27:
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
