"""csci3302_lab2 controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from math import sin, cos, atan2, pi, radians
from controller import Robot, Motor, DistanceSensor
import os

# Ground Sensor Measurements under this threshold are black
# measurements above this threshold can be considered white.
# TODO: Fill this in with a reasonable threshold that separates "line detected" from "no line detected"
#GROUND_SENSOR_THRESHOLD = 0
#didn't use this

# These are your pose values that you will update by solving the odometry equations
pose_x = 0
pose_y = 0
pose_theta = 0

# Index into ground_sensors and ground_sensor_readings for each of the 3 onboard sensors.
LEFT_IDX = 2
CENTER_IDX = 1
RIGHT_IDX = 0

# create the Robot instance.
robot = Robot()

# ePuck Constants
EPUCK_AXLE_DIAMETER = 0.053 # ePuck's wheels are 53mm apart.
EPUCK_MAX_WHEEL_SPEED = 0 # TODO: To be filled in with ePuck wheel speed in m/s
MAX_SPEED = 6.28

WHEEL_RADIUS = 0.02
AXLE_LENGTH = 0.052
RANGE = 1024 / 2


def compute_odometry(left_position_sensor, right_position_sensor):
    l = left_position_sensor
    r = right_position_sensor
    print(l)
    dl = l * WHEEL_RADIUS  # distance covered by left wheel in meter
    dr = r * WHEEL_RADIUS  # distance covered by right wheel in meter
    
    da = (dr - dl) / AXLE_LENGTH  # delta orientation

    print("Odometry: left wheel travel={0:.3f}, right wheel travel={1:.3f}, orientation={2:.3f} rad".format(dl, dr, da))
    return dl, dr, da

# get the time step of the current world.
SIM_TIMESTEP = int(robot.getBasicTimeStep())

# Initialize Motors
leftMotor = robot.getDevice('left wheel motor')
rightMotor = robot.getDevice('right wheel motor')
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))
leftMotor.setVelocity(0.0)
rightMotor.setVelocity(0.0)
left_sensor = robot.getDevice("left wheel sensor")
right_sensor = robot.getDevice("right wheel sensor")
left_sensor.enable(SIM_TIMESTEP)
right_sensor.enable(SIM_TIMESTEP)

# Initialize and Enable the Ground Sensors
gsr = [0, 0, 0]
ground_sensors = [robot.getDevice('gs0'), robot.getDevice('gs1'), robot.getDevice('gs2')]
for gs in ground_sensors:
    gs.enable(SIM_TIMESTEP)

pose_x = 0
pose_y = 0
pose_theta = 0
theta = 0

# Allow sensors to properly initialize
for i in range(10): robot.step(SIM_TIMESTEP)  

vL = 5 # TODO: Initialize variable for left speed
vR = 5 # TODO: Initialize variable for right speed

curState = 0
timer = 0
count = 0
left_sensor_offset = 0
right_sensor_offset = 0
# Main Control Loop:
while robot.step(SIM_TIMESTEP) != -1:
    
    count = count + 1

    # Read ground sensor values
    for i, gs in enumerate(ground_sensors):
        gsr[i] = gs.getValue()
    
    if curState == 0:
        #foward state
        leftMotor.setVelocity(vL*0.1)
        rightMotor.setVelocity(vR*0.1)
        if gsr[0] > 310 and gsr[2] > 310:
            timer = 0
    elif curState == 1:
        #right turn state
        leftMotor.setVelocity(vL)
        rightMotor.setVelocity(-vR)
        curState = 0
    elif curState == 2:
        #left turn state
        leftMotor.setVelocity(-vL)
        rightMotor.setVelocity(vR)
        curState = 0
    if gsr[0] < 320 and gsr[2] < 320:
        timer += 1
        if(timer > 50):
            #loop closure
            left_sensor_offset = left_sensor.getValue()
            right_sensor_offset = right_sensor.getValue()
            curState = 0
    elif gsr[0] < 310:
        curState = 2
    elif gsr[2] < 310:
        curState = 1
                  
    
    left_sensor_val = left_sensor.getValue() - left_sensor_offset
    right_sensor_val = right_sensor.getValue() - right_sensor_offset
    
    
    
    
    
    # TODO: Call update_odometry Here
    left_travel, right_travel, angle = compute_odometry(left_sensor_val, right_sensor_val)
    # Hints:
    #
    # 1) Divide vL/vR by MAX_SPEED to normalize, then multiply with
    # the robot's maximum speed in meters per second. 
    #
    # 2) SIM_TIMESTEP tells you the elapsed time per step. You need
    # to divide by 1000.0 to convert it to seconds
    #
    # 3) Do simple sanity checks. In the beginning, only one value
    # changes. Once you do a right turn, this value should be constant.
    #
    # 4) Focus on getting things generally right first, then worry
    # about calculating odometry in the world coordinate system of the
    # Webots simulator first (x points down, y points right)
    distance = (left_travel+right_travel)/2
    
    pose_x = pose_x + (distance * cos(angle))/count
    pose_y = pose_y + (distance * sin(angle))/count
    pose_theta = angle
    

    
    # TODO: Insert Loop Closure Code Here
    
    #loop closure is handled above 
 

    print("Current pose: [%5f, %5f, %5f]" % (pose_x, pose_y, pose_theta))