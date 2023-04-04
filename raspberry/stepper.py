#!/usr/bin/python3
from unittest import skip
import RPi.GPIO as GPIO
import signal
import time
import datetime
import db_read as DB

class stepper_service:
  killer = False
  def __init__(self):
    signal.signal(signal.SIGINT, self.exit_servivice)
    signal.signal(signal.SIGTERM, self.exit_servivice)

  def exit_servivice(self, *args):
    self.killer = True
    reset_status(new_status)
    exit(0)

 
in1 = 17
in2 = 18
in3 = 27
in4 = 22
 
# careful lowering this, at some point you run into the mechanical limitation of how quick your motor can move
step_sleep = 0.002
 
# defining stepper motor sequence (found in documentation http://www.4tronix.co.uk/arduino/Stepper-Motors.php)
step_sequence = [[1,0,0,1],
                 [1,0,0,0],
                 [1,1,0,0],
                 [0,1,0,0],
                 [0,1,1,0],
                 [0,0,1,0],
                 [0,0,1,1],
                 [0,0,0,1]]
 
# setting up
GPIO.setmode( GPIO.BCM )
GPIO.setup( in1, GPIO.OUT )
GPIO.setup( in2, GPIO.OUT )
GPIO.setup( in3, GPIO.OUT )
GPIO.setup( in4, GPIO.OUT )
 
# initializing
GPIO.output( in1, GPIO.LOW )
GPIO.output( in2, GPIO.LOW )
GPIO.output( in3, GPIO.LOW )
GPIO.output( in4, GPIO.LOW )
 
 
motor_pins = [in1,in2,in3,in4]

 
 
def cleanup():
    GPIO.output( in1, GPIO.LOW )
    GPIO.output( in2, GPIO.LOW )
    GPIO.output( in3, GPIO.LOW )
    GPIO.output( in4, GPIO.LOW )
    GPIO.cleanup()

# the meat

def rotate(step_count,direction):
    motor_step_counter = 0
    try:
        i = 0
        for i in range(step_count):
            for pin in range(0, len(motor_pins)):
                GPIO.output( motor_pins[pin], step_sequence[motor_step_counter][pin] )
            if direction==True:
                motor_step_counter = (motor_step_counter - 1) % 8
            elif direction==False:
                motor_step_counter = (motor_step_counter + 1) % 8
            else: # defensive programming
                print( "Direction should be either True or False" )
                cleanup()
                exit( 1 )
            time.sleep( step_sleep )
    
    except KeyboardInterrupt:
        cleanup()
        exit( 1 )


def return_status(temp : float):
    if temp > 30:
        return "hot"
    elif temp > 20:
        return "normal"
    else:
        return "cool"

def reset_status(status):
    if (status == "hot"):
        rotate(1024,True)
    elif (status == "normal"):
        rotate(512,True)
    else:
        pass

old_status = "cool"
new_status = "cool"
if __name__ == '__main__':
  instance = stepper_service()
  while not instance.killer:
    try:
        try:
            temperature = DB.get_db_temperature_c()
        except:
            print("skipped reading")
            continue
        currentDT = datetime.datetime.now()
        date_time = currentDT.strftime("%Y-%m-%d %H:%M:%S")
        print(date_time,temperature,"c")
        new_status = return_status(temperature)
        if (old_status == "cool") & (new_status == "normal"):
            rotate(512,False)
            old_status = new_status
        elif (old_status == "cool") & (new_status == "hot"):
            rotate(1024,False)
            old_status = new_status
        elif (old_status == "normal") & (new_status == "cool"):
            rotate(512,True)
            old_status = new_status
        elif (old_status == "normal") & (new_status == "hot"):
            rotate(512,False)
            old_status = new_status
        elif (old_status == "hot") & (new_status == "normal"):
            rotate(512,True)
            old_status = new_status
        elif (old_status == "hot") & (new_status == "cool"):
            rotate(1024,True)
            old_status = new_status
        else:
            old_status = new_status

        time.sleep(1)
    except KeyboardInterrupt:
        reset_status(new_status)
        cleanup()
        exit( 1 )


cleanup()
exit( 0 )