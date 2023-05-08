"""grocery controller."""

# Nov 2, 2022

from controller import Robot, Motor, Camera, RangeFinder, Lidar, Keyboard
import math
import numpy as np
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink
import ikpy.utils.plot as plot_utils
import matplotlib.pyplot
from mpl_toolkits.mplot3d import Axes3D
import cv2
import ipywidgets as widgets

#Initialization
print("=== Initializing Grocery Shopper...\n")
#Consts
MAX_SPEED = 7.0  # [rad/s]
MAX_SPEED_MS = 0.633 # [m/s]
AXLE_LENGTH = 0.4044 # m
MOTOR_LEFT = 10
MOTOR_RIGHT = 11
N_PARTS = 12
LIDAR_ANGLE_BINS = 667
LIDAR_SENSOR_MAX_RANGE = 5.5 # Meters
LIDAR_ANGLE_RANGE = math.radians(240)

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# The Tiago robot has multiple motors, each identified by their names below
part_names = ("head_2_joint", "head_1_joint", "torso_lift_joint", "arm_1_joint",
              "arm_2_joint",  "arm_3_joint",  "arm_4_joint",      "arm_5_joint",
              "arm_6_joint",  "arm_7_joint",  "wheel_left_joint", "wheel_right_joint",
              "gripper_left_finger_joint","gripper_right_finger_joint")


# All motors except the wheels are controlled by position control. The wheels
# are controlled by a velocity controller. We therefore set their position to infinite.
target_pos = (0.0, 0.0, 0.35, 0.07, 1.02, -3.16, 1.27, 1.32, 0.0, 1.41, 'inf', 'inf',0.045,0.045)

robot_parts={}
for i, part_name in enumerate(part_names):
    robot_parts[part_name]=robot.getDevice(part_name)
    robot_parts[part_name].setPosition(float(target_pos[i]))
    robot_parts[part_name].setVelocity(robot_parts[part_name].getMaxVelocity() / 2.0)
    position_sensor = robot_parts[part_name].getPositionSensor()
    position_sensor.enable(timestep)

# Enable gripper encoders (position sensors)
left_gripper_enc=robot.getDevice("gripper_left_finger_joint_sensor")
right_gripper_enc=robot.getDevice("gripper_right_finger_joint_sensor")
left_gripper_enc.enable(timestep)
right_gripper_enc.enable(timestep)

# Enable Camera
camera = robot.getDevice('camera')
camera.enable(timestep)
camera.recognitionEnable(timestep)

# Enable GPS and compass localization
gps = robot.getDevice("gps")
gps.enable(timestep)
compass = robot.getDevice("compass")
compass.enable(timestep)

# Enable LiDAR
lidar = robot.getDevice('Hokuyo URG-04LX-UG01')
lidar.enable(timestep)
lidar.enablePointCloud()

# Enable display
display = robot.getDevice("display")

# Odometry
pose_x     = 0
pose_y     = 0
pose_theta = 0

vL = 0
vR = 0

lidar_sensor_readings = [] # List to hold sensor readings
lidar_offsets = np.linspace(-LIDAR_ANGLE_RANGE/2., +LIDAR_ANGLE_RANGE/2., LIDAR_ANGLE_BINS)
lidar_offsets = lidar_offsets[83:len(lidar_offsets)-83] # Only keep lidar readings not blocked by robot chassis

map = None

# We are using a keyboard to remote control the robot
keyboard = robot.getKeyboard()
keyboard.enable(timestep)

print("=== Initialization Complete!\n")


###############
# IKPY setup
###############
print("=== Setting up IKPY...\n")

# get the URDF file **UNCOMMENT AND RUN ONLY ONCE TO INITIALIZE**
# with open("tiago_urdf.urdf", "w") as file:  
    # file.write(robot.getUrdf())

# create chain for the robot
base_elements=["base_link", "base_link_Torso_joint", "Torso", "torso_lift_joint", "torso_lift_link", 
               "torso_lift_link_TIAGo front arm_11367_joint", "TIAGo front arm_11367"]

# include finger link (end-effector) to chain
my_chain = Chain.from_urdf_file("robot_urdf.urdf", last_link_vector=[0.004, 0, -0.1741], 
            base_elements=["base_link", "base_link_Torso_joint", "Torso", "torso_lift_joint", "torso_lift_link", 
                           "torso_lift_link_TIAGo front arm_11367_joint", "TIAGo front arm_11367"])
                                    
for link_id in range(len(my_chain.links)):
    # link obj
    link = my_chain.links[link_id]
    # ignore "torso_lift_joint"
    if link.name not in part_names or  link.name == "torso_lift_joint":
        my_chain.active_links_mask[link_id] = False
        
# Initialize the arm motors and encoders.
motors = []
for link in my_chain.links:
    if link.name in part_names and link.name != "torso_lift_joint":
        motor = robot.getDevice(link.name)

        # Make sure to account for any motors that
        # require a different maximum velocity!
        if link.name == "torso_lift_joint":
            motor.setVelocity(0.07)
        else:
            motor.setVelocity(0.5)
            
        position_sensor = motor.getPositionSensor()
        position_sensor.enable(timestep)
        motors.append(motor)
                          
print(my_chain.links)

# disable torso lift joint
# my_chain.active_links_mask[1] = False

# get intial position **GIVES NAN VALUES ON FIRST RUN -- RUN TWICE**
initial_position = [0,0,0,0] + [m.getPositionSensor().getValue() for m in motors] + [0,0,0,0]
print(initial_position)

print("=== IKPY Setup Complete!\n")

# ------------------------------------------------------------------
# Helper Functions

mode = "manual"

# open grippers
robot_parts["gripper_left_finger_joint"].setPosition(0.045)
robot_parts["gripper_right_finger_joint"].setPosition(0.045)
gripper_status="open"

# set torso_lift_joint to 0
robot.getDevice("torso_lift_joint").setPosition(0.25)
error = 0
c = 0

# Main Loop
while robot.step(timestep) != -1:
    
    ###################
    #
    # Controller
    #
    ###################
    if mode == "manual":
        key = keyboard.getKey()
        while(keyboard.getKey() != -1): pass
        if key == ord('A'):
            vL = -MAX_SPEED
            vR = MAX_SPEED
        elif key == ord('D'):
            vL = MAX_SPEED
            vR = -MAX_SPEED
        elif key == ord('W'):
            vL = MAX_SPEED
            vR = MAX_SPEED
        elif key == ord('S'):
            vL = -MAX_SPEED
            vR = -MAX_SPEED
        # elif key == keyboard.LEFT:
          
        elif key == keyboard.RIGHT:
            if(gripper_status=="open"):
                # Close gripper, note that this takes multiple time steps...
                robot_parts["gripper_left_finger_joint"].setPosition(0)
                robot_parts["gripper_right_finger_joint"].setPosition(0)
                gripper_status="closed"
            else:
                ## Open gripper
                robot_parts["gripper_left_finger_joint"].setPosition(0.045)
                robot_parts["gripper_right_finger_joint"].setPosition(0.045)
                gripper_status="open"
        elif key == keyboard.UP:
            # get intial position, disabled links have position '0'
            initial_position = [0,0,0,0] + [m.getPositionSensor().getValue() for m in motors] + [0,0,0,0]
            # get yellow obj position relative to the camera
            target = yellowobjsGPS[0]
            print(target)
            
            # adjust target from camera coordinate to be realtive to the end effector
            target = [-target[2] + 0.47, -target[0] + 0.85, target[1] + 0.895]
            # ik operation
            ikResults = my_chain.inverse_kinematics(target, target_orientation = [0,0,1], orientation_mode="Y")
            # apply ik to robot's end effector
            for res in range(len(ikResults)):
                # ignore non-controllable links
                if my_chain.links[res].name in part_names:
                    if my_chain.links[res].name == "torso_lift_joint":
                        robot.getDevice(my_chain.links[res].name).setPosition(0.25)
                        print("Setting {} to {}".format(my_chain.links[res].name, 0.25))
                    elif my_chain.links[res].name == "gripper_right_finger_joint":
                        robot.getDevice(my_chain.links[res].name).setPosition(0.045)
                        print("Setting {} to {}".format(my_chain.links[res].name, 0.045))
                    else:
                        robot.getDevice(my_chain.links[res].name).setPosition(ikResults[res])
                        print("Setting {} to {}".format(my_chain.links[res].name, ikResults[res]))
            
        elif key == keyboard.DOWN:
            # get intial position, disabled links have position '0'
            initial_position = [0,0,0,0] + [m.getPositionSensor().getValue() for m in motors] + [0,0,0,0]
            # send arm to directly above basket
            target = [0.3, 0.0, 0.8]
            print(target)

            # ik operation
            ikResults = my_chain.inverse_kinematics(target, target_orientation = [0,0,1], orientation_mode="Y")
            # apply ik to robot's end effector
            for res in range(len(ikResults)):
                # ignore non-controllable links
                if my_chain.links[res].name in part_names:
                    if my_chain.links[res].name == "torso_lift_joint":
                        robot.getDevice(my_chain.links[res].name).setPosition(0.25)
                        print("Setting {} to {}".format(my_chain.links[res].name, 0.25))
                    elif my_chain.links[res].name == "gripper_right_finger_joint":
                        robot.getDevice(my_chain.links[res].name).setPosition(0.0)
                        print("Setting {} to {}".format(my_chain.links[res].name, 0.0))
                    else:
                        robot.getDevice(my_chain.links[res].name).setPosition(ikResults[res])
                        print("Setting {} to {}".format(my_chain.links[res].name, ikResults[res]))
        elif key == ord('I'):
            # ik operation
            # adjust target manually (teleoperation)
            target = [target[2] + 0.01, target[0] + 0.01, target[1] + 0.01]
            # ik operation
            ikResults = my_chain.inverse_kinematics(target, target_orientation = [0,0,1], orientation_mode="Y")
            # apply ik to robot's end effector
            for res in range(len(ikResults)):
                # ignore non-controllable links
                if my_chain.links[res].name in part_names:
                    if my_chain.links[res].name == "torso_lift_joint":
                        robot.getDevice(my_chain.links[res].name).setPosition(0.25)
                        print("Setting {} to {}".format(my_chain.links[res].name, 0.25))
                    elif my_chain.links[res].name == "gripper_right_finger_joint":
                        robot.getDevice(my_chain.links[res].name).setPosition(0.045)
                        print("Setting {} to {}".format(my_chain.links[res].name, 0.045))
                    else:
                        robot.getDevice(my_chain.links[res].name).setPosition(ikResults[res])
                        print("Setting {} to {}".format(my_chain.links[res].name, ikResults[res]))
        elif key == ord('J'):
            # ik operation
            # adjust target manually (teleoperation)
            target = [target[2] - 0.01, target[0] - 0.01, target[1] - 0.01]
            # ik operation
            ikResults = my_chain.inverse_kinematics(target, target_orientation = [0,0,1], orientation_mode="Y")
            # apply ik to robot's end effector
            for res in range(len(ikResults)):
                # ignore non-controllable links
                if my_chain.links[res].name in part_names:
                    if my_chain.links[res].name == "torso_lift_joint":
                        robot.getDevice(my_chain.links[res].name).setPosition(0.25)
                        print("Setting {} to {}".format(my_chain.links[res].name, 0.25))
                    elif my_chain.links[res].name == "gripper_right_finger_joint":
                        robot.getDevice(my_chain.links[res].name).setPosition(0.045)
                        print("Setting {} to {}".format(my_chain.links[res].name, 0.045))
                    else:
                        robot.getDevice(my_chain.links[res].name).setPosition(ikResults[res])
                        print("Setting {} to {}".format(my_chain.links[res].name, ikResults[res]))
        elif key == ord('K'):
            # ik operation
            # adjust target manually (teleoperation)
            target = [target[2] + 0.0, target[0] + 0.0, target[1] + 0.01]
            # ik operation
            ikResults = my_chain.inverse_kinematics(target, target_orientation = [0,0,1], orientation_mode="Y")
            # apply ik to robot's end effector
            for res in range(len(ikResults)):
                # ignore non-controllable links
                if my_chain.links[res].name in part_names:
                    if my_chain.links[res].name == "torso_lift_joint":
                        robot.getDevice(my_chain.links[res].name).setPosition(0.25)
                        print("Setting {} to {}".format(my_chain.links[res].name, 0.25))
                    elif my_chain.links[res].name == "gripper_right_finger_joint":
                        robot.getDevice(my_chain.links[res].name).setPosition(0.045)
                        print("Setting {} to {}".format(my_chain.links[res].name, 0.045))
                    else:
                        robot.getDevice(my_chain.links[res].name).setPosition(ikResults[res])
                        print("Setting {} to {}".format(my_chain.links[res].name, ikResults[res]))
        elif key == ord('L'):
            # ik operation
            # adjust target manually (teleoperation)
            target = [target[2] + 0.0, target[0] + 0.0, target[1] - 0.01]
            # ik operation
            ikResults = my_chain.inverse_kinematics(target, target_orientation = [0,0,1], orientation_mode="Y")
            # apply ik to robot's end effector
            for res in range(len(ikResults)):
                # ignore non-controllable links
                if my_chain.links[res].name in part_names:
                    if my_chain.links[res].name == "torso_lift_joint":
                        robot.getDevice(my_chain.links[res].name).setPosition(0.25)
                        print("Setting {} to {}".format(my_chain.links[res].name, 0.25))
                    elif my_chain.links[res].name == "gripper_right_finger_joint":
                        robot.getDevice(my_chain.links[res].name).setPosition(0.045)
                        print("Setting {} to {}".format(my_chain.links[res].name, 0.045))
                    else:
                        robot.getDevice(my_chain.links[res].name).setPosition(ikResults[res])
                        print("Setting {} to {}".format(my_chain.links[res].name, ikResults[res]))
        else: # slow down
            vL *= 0.75
            vR *= 0.75
            
    num_objects = camera.getRecognitionNumberOfObjects()
    obj = camera.getRecognitionObjects()
    yellowobjsGPS = []
    for i in range(num_objects):
        if (obj[i].getColors() == [1.0, 1.0, 0]):
            yellowobjsGPS.append(obj[i].getPosition())
             
    # print(yellowobjsGPS) 
                  
    robot_parts["wheel_left_joint"].setVelocity(vL)
    robot_parts["wheel_right_joint"].setVelocity(vR)
