import cv2
import dlib
import os
import smtplib, ssl
from email.mime.multipart import MIMEMultipart
from email import encoders
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
import numpy as np
import time
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from picamera.array import PiRGBArray
from picamera import PiCamera
import RPi.GPIO as GPIO
from smbus2 import SMBus
from mlx90614 import MLX90614

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

servoxPIN = 32
servoyPIN = 33

camera = PiCamera()
# camera.resolution = (640, 480)
camera.framerate = 32
camera.rotation = 180
rawCapture = PiRGBArray(camera)
# allow the camera to warmup
time.sleep(0.1)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

model = load_model('models/mask_detector.h5')

mask_detection_completed = False
mask_count = 0

temperature_check_completed = False

send_email = os.environ.get('SEND_EMAIL')
if send_email == 'TRUE':
    sender_email = os.environ.get('EMAIL_ID')
    receiver_email = os.environ.get('EMAIL_ID')
    password = os.environ.get('EMAIL_PWD')

    message = MIMEMultipart("alternative")
    message["Subject"] = "Alert: A New Person Entered the Premises"
    message["From"] = sender_email
    message["To"] = receiver_email


def detect_mask(image):
    copy_img = image.copy()

    resized = cv2.resize(copy_img, (254, 254))

    resized = img_to_array(resized)
    resized = preprocess_input(resized)

    resized = np.expand_dims(resized, axis=0)

    mask, _ = model.predict([resized])[0]

    return mask


def email(img_path, temp, mask):
    with open(img_path, 'rb') as f:
        # set attachment mime and file name, the image type is png
        mime = MIMEBase('image', 'png', filename='img1.png')
        # add required header data:
        mime.add_header('Content-Disposition', 'attachment', filename='img1.png')
        mime.add_header('X-Attachment-Id', '0')
        mime.add_header('Content-ID', '<0>')
        # read attachment file content into the MIMEBase object
        mime.set_payload(f.read())
        # encode with base64
        encoders.encode_base64(mime)
        # add MIMEBase object to MIMEMultipart object
        message.attach(mime)

    body = MIMEText('''
    <html>
        <body>
            <h1>Alert</h1>
            <h2>A new has Person entered the Premises</h2>
            <h2>Body Temperature: {}</h2>
            <h2>Mask: {}</h2>
            <h2>Time: {}</h2>
            <p>
                <img src="cid:0">
            </p>
        </body>
    </html>'''.format(temp, mask, datetime.now()), 'html', 'utf-8')

    # Add HTML/plain-text parts to MIMEMultipart message
    # The email client will try to render the last part first
    message.attach(body)

    # Create secure connection with server and send email
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(
            sender_email, receiver_email, message.as_string()
        )


class Servo:
    def __init__(self, pin):
        self.pin = pin

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.pin, GPIO.OUT)

        self.servo = GPIO.PWM(self.pin, 50)
        self.servo.start(2.5)  # Initialization

    def setAngle(self, angle):
        duty = angle / 18 + 2
        GPIO.output(self.pin, True)
        self.servo.ChangeDutyCycle(duty)
        time.sleep(0.2)
        GPIO.output(self.pin, False)
        self.servo.ChangeDutyCycle(0)

    def reset(self):
        GPIO.output(self.pin, True)
        self.servo.ChangeDutyCycle(2.5)
        time.sleep(1)
        GPIO.output(self.pin, False)
        self.servo.ChangeDutyCycle(0)

    def stop(self):
        self.servo.stop()


class PID:
    def __init__(self, kP=1, kI=0, kD=0):
        # initialize gains
        self.kP = kP
        self.kI = kI
        self.kD = kD

    def initialize(self):
        # intialize the current and previous time
        self.currTime = time.time()
        self.prevTime = self.currTime

        # initialize the previous error
        self.prevError = 0

        # initialize the term result variables
        self.cP = 0
        self.cI = 0
        self.cD = 0

    def update(self, error, sleep=0.2):
        # pause for a bit
        time.sleep(sleep)

        # grab the current time and calculate delta time
        self.currTime = time.time()
        deltaTime = self.currTime - self.prevTime

        # delta error
        deltaError = error - self.prevError

        # proportional term
        self.cP = error

        # integral term
        self.cI += error * deltaTime

        # derivative term and prevent divide by zero
        self.cD = (deltaError / deltaTime) if deltaTime > 0 else 0

        # save previous time and error for the next update
        self.prevTime = self.currTime
        self.prevError = error

        # sum the terms and return
        return sum([
            self.kP * self.cP,
            self.kI * self.cI,
            self.kD * self.cD])

    def reset(self):
        self.currTime = time.time()
        self.prevTime = self.currTime

        # initialize the previous error
        self.prevError = 0

        # initialize the term result variables
        self.cP = 0
        self.cI = 0
        self.cD = 0


servoX = Servo(servoxPIN)
servoY = Servo(servoyPIN)

servoX.setAngle(90)
servoY.setAngle(90)

bus = SMBus(1)
sensor = MLX90614(bus, address=0x5A)
temp = None

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    img = frame.array
    ret = True

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if ret:

        if mask_detection_completed is False:
            mask_prob = detect_mask(img)

            if mask_prob > 0.5:
                mask_count += 1

                if mask_count >= 5:
                    mask_detection_completed = True
                    pidX = PID()
                    pidY = PID()

                    pidX.initialize()
                    pidY.initialize()

            elif mask_prob < 0.5:
                cv2.putText(img, 'No Mask Detected', (200, 200), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 3)

        elif mask_detection_completed:
            if temperature_check_completed is False:
                cv2.putText(img, 'Mask Detected', (200, 200), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 0), 3)

                faces = detector(img_gray, 0)

                if len(faces) > 0:

                    for face in faces:

                        landmarks = predictor(img_gray, face)

                        # unpack the 68 landmark coordinates from the dlib object into a list
                        landmarks_list = []
                        for i in range(0, landmarks.num_parts):
                            landmarks_list.append((landmarks.part(i).x, landmarks.part(i).y))

                        dist = np.sqrt((landmarks.part(21).x - landmarks.part(22).x) ** 2 + (
                                landmarks.part(21).y - landmarks.part(22).y) ** 2)

                        face_ptx, face_pty = (int((landmarks.part(21).x + landmarks.part(22).x) / 2),
                                              int((landmarks.part(21).y + landmarks.part(22).y) / 2) - int(dist))

                        cv2.circle(img, (landmarks.part(21).x, landmarks.part(21).y), 4, (255, 255, 255), -1)
                        cv2.circle(img, (landmarks.part(22).x, landmarks.part(22).y), 4, (255, 255, 255), -1)
                        cv2.circle(img, (face_ptx, face_pty), 4, (0, 255, 0), -1)

                        Y, X, _ = img.shape

                        sensor_ptx, sensor_pty = (int(X / 2), int(Y / 3))

                        cv2.circle(img, (sensor_ptx, sensor_pty), 5, (255, 0, 0), -1)

                        diff_x, diff_y = sensor_ptx - face_ptx, sensor_pty - face_pty
                        cv2.putText(img, f'Distance: {diff_x}, {diff_y}', (0, 500),
                                    cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 5)

                        if -10 < diff_x < 10 and -10 < diff_y < 10:
                            temp = sensor.get_amb_temp()
                            cv2.putText(img, 'Body Temp: {}F '.format(temp), (200, 400),
                                        cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 5)

                            cv2.putText(img, 'Please Proceed! ', (200, 600),
                                        cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 5)
                            servoX.reset()
                            servoY.reset()
                            temperature_check_completed = True
                        else:
                            servoX.setAngle(-1 * pidX.update(diff_x))
                            servoY.setAngle(-1 * pidY.update(diff_x))
                            if diff_x > 0:
                                cv2.putText(img, '->', (800, 500),
                                            cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 5)
                            elif diff_x < 0:
                                cv2.putText(img, '<-', (800, 500),
                                            cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 5)

                            if diff_y > 0:
                                cv2.putText(img, 'v', (700, 500),
                                            cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 5)
                            elif diff_y < 0:
                                cv2.putText(img, '^', (700, 500),
                                            cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 5)
                else:
                    cv2.putText(img, 'No Face Detected Please Remove Mask ', (0, 500),
                                cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 5)

            elif temperature_check_completed:
                c_id = os.environ.get('COUNTER_ID')
                cv2.imwrite('pictures/{}.jpg'.format(str(c_id)), img)
                email('pictures/{}.jpg'.format(c_id), temp, 'Wearing')
                os.environ['COUNTER_ID'] = str(int(c_id) + 1)

                # Reset
                mask_detection_completed = False
                mask_count = 0
                temperature_check_completed = False
                temp = None

        cv2.imshow('window', img)

    rawCapture.truncate(0)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break
    elif key == ord('a'):
        cv2.imwrite('my_pic.jpg', img)

cv2.destroyAllWindows()

servoX.stop()
servoY.stop()

GPIO.cleanup()

bus.close()

