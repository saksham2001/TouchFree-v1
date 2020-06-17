import cv2
import dlib
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import RPi.GPIO as GPIO
from smbus2 import SMBus
from mlx90614 import MLX90614

servoxPIN = 32
servoyPIN = 33

GPIO.setmode(GPIO.BOARD)

camera = PiCamera()
# camera.resolution = (640, 480)
camera.framerate = 32
camera.rotation = 0
rawCapture = PiRGBArray(camera)
# allow the camera to warmup
time.sleep(0.1)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")


class Servo:
    def __init__(self, pin):
        self.pin = pin

        GPIO.setup(self.pin, GPIO.OUT)

        self.servo = GPIO.PWM(self.pin, 50)
        self.servo.start(0)  # Initialization

    def setAngle(self, angle):
        duty = ((90 + angle) / 18) + 2
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

pidX = PID()
pidY = PID()

pidX.initialize()
pidY.initialize()

bus = SMBus(1)
sensor = MLX90614(bus, address=0x5A)
temp = None

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    img = frame.array
    ret = True

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if ret:
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

                face_ptx, face_pty = (int((landmarks.part(21).x + landmarks.part(22).x)/2),
                                      int((landmarks.part(21).y + landmarks.part(22).y)/2) - int(dist))

                cv2.circle(img, (landmarks.part(21).x, landmarks.part(21).y), 2, (255, 255, 255), -1)
                cv2.circle(img, (landmarks.part(22).x, landmarks.part(22).y), 2, (255, 255, 255), -1)
                cv2.circle(img, (face_ptx, face_pty), 4, (0, 255, 0), -1)

                Y, X, _ = img.shape

                sensor_ptx, sensor_pty = (int(X/2), int(Y/3))

                cv2.circle(img, (sensor_ptx, sensor_pty), 3, (255, 0, 0), -1)

                diff_x, diff_y = sensor_ptx-face_ptx, sensor_pty-face_pty
                cv2.putText(img, 'Distance: {}, {}'.format(diff_x, diff_y), (0, 200),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
                if -10 < diff_x < 10 and -10 < diff_y < 10:
                    temp = sensor.get_amb_temp()
                    cv2.putText(img, 'Body Temp: 98.6F ', (200, 400),
                                cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 5)

                    if temp > 100:
                        cv2.putText(img, 'Body Temperature too High! ', (200, 600),
                                    cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 5)
                        ret = False
                    else:
                        cv2.putText(img, 'Please Proceed! ', (200, 600),
                                    cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 5)
                        servoX.reset()
                        servoY.reset()
                        temperature_check_completed = True
                        ret = False
                else:
                    servoX.setAngle(-1 * pidX.update(diff_x))
                    servoY.setAngle(-1 * pidY.update(diff_x))
                    if diff_x > 0:
                        cv2.putText(img, '->', (200, 200),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 3)
                    elif diff_x < 0:
                        cv2.putText(img, '<-', (200, 200),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 3)
                    if diff_y > 0:
                        cv2.putText(img, 'v', (400, 400),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 3)
                    elif diff_y < 0:
                        cv2.putText(img, '^', (400, 400),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 3)

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


