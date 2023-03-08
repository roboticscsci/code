"""lab3 controller."""
# Copyright University of Colorado Boulder 2022
# CSCI 3302 "Introduction to Robotics" Lab 3 Base Code.

from controller import Robot, Motor
import math

# TODO: Fill out with correct values from Robot Spec Sheet (or inspect PROTO definition for the robot)
MAX_SPEED = 2.84 # [rad/s]
MAX_SPEED_MS = 0.22 # [m/s]
AXLE_LENGTH = 0.16 # [m]



MOTOR_LEFT = 0 # Left wheel index
MOTOR_RIGHT = 1 # Right wheel index

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# The Turtlebot robot has two motors
part_names = ("left wheel motor", "right wheel motor")


# Set wheels to velocity control by setting target position to 'inf'
# You should not use target_pos for storing waypoints. Leave it unmodified and 
# use your own variable to store waypoints leading up to the goal
target_pos = ('inf', 'inf') 
robot_parts = []

for i in range(len(part_names)):
        robot_parts.append(robot.getDevice(part_names[i]))
        robot_parts[i].setPosition(float(target_pos[i]))

# Odometry
pose_x     = 0
pose_y     = 0
pose_theta = 0
    
    
# Rotational Motor Velocity [rad/s]
vL = 1
vR = 1

# TODO
# Create you state and goals (waypoints) variable here
# You have to MANUALLY figure out the waypoints, one sample is provided for you in the instructions
goals = [[5.5, -2.2, 0], [6, 3.5, 2], [5, 10, 4]]
current_goal = goals[0]

while robot.step(timestep) != -1:
    print(current_goal)
    # STEP 2.1: Calculate error with respect to current and goal position
    dx = current_goal[0]-pose_x
    dy = current_goal[1]-pose_y
    posError = math.sqrt(dx**2 + dy**2)
    theta = math.atan2(dy, dx)
    bearingError = theta - pose_theta
    headingError = current_goal[2] - pose_theta
    
 

    #checks if the poserror is low or if its at the green square
    if(posError < 0.5 or (pose_x >= 1.4 and pose_x <=1.45 and pose_y >= 8.5 and pose_y <= 9.0)):
        goals.pop(0)
        if(len(goals) > 0):
            current_goal = goals[0]
        else:
            robot_parts[MOTOR_LEFT].setVelocity(0)
            robot_parts[MOTOR_RIGHT].setVelocity(0)
            break
    
    # STEP 2.2: Feedback Controller
    p1 = .5
    p2 = 4
    p3 = 3
    dX = p1 * posError
    dTheta = p2*bearingError + p3*headingError
    


    # STEP 1: Inverse Kinematics Equations (vL and vR as a function dX and dTheta)
    
    
    vL = dX - (dTheta * AXLE_LENGTH) / 2
    vR = dX + (dTheta * AXLE_LENGTH) / 2

    
    # STEP 2.3: Proportional velocities
    #implimented in feedback controller
    

    # STEP 2.4: Clamp wheel speeds
    vL = min(MAX_SPEED, max(-MAX_SPEED, vL))
    vR = min(MAX_SPEED, max(-MAX_SPEED, vR))

    #updating odometry
    distL = vL/MAX_SPEED * MAX_SPEED_MS * timestep / 1000
    distR = vR/MAX_SPEED * MAX_SPEED_MS * timestep / 1000
    pose_x += (distL+distR)/2 * math.cos(pose_theta)
    pose_y += (distL+distR)/2 * math.sin(pose_theta)
    pose_theta += (distR-distL)/AXLE_LENGTH
    print(pose_x, pose_y, pose_theta)
    

    
    

    ########## End Odometry Code ##################
    
    ########## Do not change ######################
    # Bound pose_theta between [-pi, 2pi+pi/2]
    # Important to not allow big fluctuations between timesteps (e.g., going from -pi to pi)
    if pose_theta > 6.28+3.14/2: pose_theta -= 6.28
    if pose_theta < -3.14: pose_theta += 6.28
    ###############################################

    # Set robot motors to the desired velocities
    robot_parts[MOTOR_LEFT].setVelocity(vL)
    robot_parts[MOTOR_RIGHT].setVelocity(vR)
    