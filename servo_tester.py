import RPi.GPIO as GPIO
from time import sleep

pinX = 32
pinY = 33

GPIO.setmode(GPIO.BOARD)

GPIO.setup(pinX, GPIO.OUT)
GPIO.setup(pinY, GPIO.OUT)


servoX = GPIO.PWM(pinX, 50)
servoY = GPIO.PWM(pinY, 50)

servoX.start(0)
servoY.start(0)


def SetAngle(pwm, pin, angle):
    duty = angle / 18 + 2
    GPIO.output(pin, True)
    pwm.ChangeDutyCycle(duty)
    sleep(1)
    GPIO.output(pin, False)
    pwm.ChangeDutyCycle(0)


for i in range(0, 180, 10):
    SetAngle(servoX, pinX, i)
    sleep(0.5)
    SetAngle(servoY, pinY, i)
    sleep(0.5)

servoX.stop()
servoY.stop()

GPIO.cleanup()


