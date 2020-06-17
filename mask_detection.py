from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
import time

model = load_model('models/mask_detector.h5')
cap = cv2.VideoCapture(0)

camera = PiCamera()
camera.rotation = 180
# camera.framerate = 32
rawCapture = PiRGBArray(camera)
# allow the camera to warmup
time.sleep(0.1)


def detect_mask(image):
    copy_img = image.copy()

    resized = cv2.resize(copy_img, (254, 254))

    resized = img_to_array(resized)
    resized = preprocess_input(resized)

    resized = np.expand_dims(resized, axis=0)

    mask, _ = model.predict([resized])[0]

    return mask


for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    img = frame.array
    ret = True

    if ret:

        mask_prob = detect_mask(img)

        if mask_prob > 0.5:
            cv2.putText(img, 'Mask Detected', (200, 200), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 0), 3)

        elif mask_prob < 0.5:
            cv2.putText(img, 'No Mask Detected', (200, 200), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 3)

        cv2.imshow('window', img)

    else:
        cv2.imshow('window', img)
	
    rawCapture.truncate(0)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break
    elif key == ord('a'):
        cv2.imwrite('my_pic.jpg', img)


cv2.destroyAllWindows()

