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

def compute_odometry(left_position_sensor, right_position_sensor):

    # Purpose: To find the current position of the robot using the wheel positions
    # Inputs: left and right position sensor values

    l = left_position_sensor
    r = right_position_sensor
    dl = l * WHEEL_RADIUS  # distance covered by left wheel in meter
    dr = r * WHEEL_RADIUS  # distance covered by right wheel in meter

    # Formula to find delta orientation
    da = ((dr - dl) / AXLE_LENGTH) - np.pi/2  # delta orientation

    # returns total distance covered by left wheel (dl), right wheel (dr) and total theta from start (da)
    return dl, dr, da

def turn(robot, robot_parts, MAX_SPEED, dir):

    # Purpose: To turn the robot a certain direction
    # Inputs: robot, parts list, speed and direction (left or right)

    if dir == "left":
        print("turning left")
        robot_parts["wheel_right_joint"].setVelocity(-MAX_SPEED)
        for i in range(2):
            robot.step(timestep)
        robot_parts["wheel_right_joint"].setVelocity(MAX_SPEED)

    elif dir == "right":
        print("turning right")
        robot_parts["wheel_left_joint"].setVelocity(-MAX_SPEED)
        for i in range(2):
            robot.step(timestep)
        robot_parts["wheel_left_joint"].setVelocity(MAX_SPEED)


def calc_d_and_theta(from_nd, to_nd):
            
            # Purpose: To calculate the distance and angle from x-axis between two points
            # Inputs: From point and to point (tuples)

            dx = to_nd[1] - from_nd[1]
            dy = to_nd[0] - from_nd[0]
            d = heuristic(from_nd, to_nd)
            theta = np.arctan2(dy, dx)
            return d, theta

def heuristic(a, b):

    # Purpose: To calculate distance between two points
    # Inputs: From point and to point (tuples)

    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

if mode == "mapper":

    # Default speed set to stationary initially
    vL = 0
    vR = 0

    # Initializing map and incr variables to construct map array
    map = np.zeros(shape=[28*res,14*res])
    incr = 0.02
    res = 50

    # Choose between automatic "auto" or manual "manual" control setting for mapping
    ctrl_mode = "auto" 

    # Initializing more variables
    count = 0
    L_prev = 0
    R_prev = 0

    # Looping through to get readings
    while robot.step(timestep) != -1:

        count += 1

        # Reading lidar sensor values and trimming unwanted values
        lidar_sensor_readings = lidar.getRangeImage()
        lidar_sensor_readings = lidar_sensor_readings[83:len(lidar_sensor_readings) - 83]

        # Iterate through remaining readings individually to plot point
        for i, rho in enumerate(lidar_sensor_readings):
            alpha = lidar_offsets[i]

            # No object in sight in this angle
            if rho > LIDAR_SENSOR_MAX_RANGE:
                continue

            # The Webots coordinate system doesn't match the robot-centric axes we're used to
            rx = math.cos(alpha) * rho
            ry = -math.sin(alpha) * rho

            t = pose_theta + np.pi/2
            # Convert detection from robot coordinates into world coordinates
            wx = math.cos(t) * rx - math.sin(t) * ry + pose_x
            wy = math.sin(t) * rx + math.cos(t) * ry + pose_y

            # Ensure it is within world bounds
            if wx >= 14:
                wx = 13.999
            elif wx <= -14:
                wx = -13.999
            if wy >= 7:
                wy = 6.999
            elif wy <= -7:
                wy = -6.999
            if rho < LIDAR_SENSOR_MAX_RANGE:
                pixel_x = res*14 - int(wx*res)
                pixel_y = res*7 - int(wy*res)

                # Object detected, add to map array as value
                if map[pixel_x, pixel_y] < 1-incr:
                    map[pixel_x, pixel_y] = map[pixel_x, pixel_y] + incr

                # Convert to grayscale value
                g = map[pixel_x, pixel_y]
                gray_val = 255*(256**2*g + 256*g + g)

                # Draw on display device
                display.setColor(int(gray_val))
                display.drawPixel(pixel_x, pixel_y)

            # Drawing robot's current position in red
            display.setColor(int(0xFF0000))
            display.drawPixel(res*14 - int(pose_x*res), res*7 - int(pose_y*res))


        key = keyboard.getKey()
        while (keyboard.getKey() != -1): pass
        if ctrl_mode == "manual":
            # Keyboard input for manual mapping
            if key == keyboard.LEFT:
                vL = -MAX_SPEED
                vR = MAX_SPEED
            elif key == keyboard.RIGHT:
                vL = MAX_SPEED
                vR = -MAX_SPEED
            elif key == keyboard.UP:
                vL = MAX_SPEED
                vR = MAX_SPEED
            elif key == keyboard.DOWN:
                vL = -MAX_SPEED
                vR = -MAX_SPEED
            elif key == ord(' '):
                vL = 0
                vR = 0
            else:  # Slow down
                vL = 0
                vR = 0
        elif ctrl_mode == "auto": 
            # automatic mapping

            # Get lidar readings and split into 2nd and third quarter for left or right side
            arr_sz = len(lidar_sensor_readings)
            quart_ind = int(arr_sz/4)
            left = min(lidar_sensor_readings[quart_ind:2*quart_ind])
            right = min(lidar_sensor_readings[2*quart_ind+1:3*quart_ind])

            # If less than 2 meters away, initiate turn
            if left < 2:
                turn(robot, robot_parts, MAX_SPEED, "left")
            elif right < 2:
                turn(robot, robot_parts, MAX_SPEED, "right")
            else:
                # Else just keep swimmin'
                vL = MAX_SPEED
                vR = MAX_SPEED
        if key == ord('P'):
            # Printing the lidar output for debugging purpose
            print(lidar_sensor_readings)
            print("---------------------------------------------")
        elif key == ord('S'):

            # Saving a filtered map with threshold of 0.4

            mapFiltered = np.multiply(map > 0.4, 1)
            np.save("map.npy", mapFiltered)
            print("Map file saved")

        elif key == ord('L'):
            # Load a map
            map = np.load(str(path_to_map))
            print("Map loaded")

        elif key == ord('O'):
            # Print current pose
            print(pose_x, ", ", pose_y, ", ", pose_theta)

        odo_mode = "gps"

        # Using gps position for accuracy purposes
        pose_x = gps.getValues()[0]
        pose_y = gps.getValues()[1]

        # Getting sensor values and feeding into odometry function
        left_pos = wheel_left_sensor.getValue()
        right_pos = wheel_right_sensor.getValue()
        L_new, R_new, pose_theta = compute_odometry(left_pos, right_pos)

        ### We would normally do the following code for pose_x and pose_y but it is not accurate enough for mapping
        if odo_mode != "gps":
            d_left = L_new - L_prev
            d_right = R_new - R_prev

            distance = (d_left + d_right)/2
            pose_x += (distance * math.cos(pose_theta))
            pose_y += (distance * math.sin(pose_theta))

            L_prev = L_new
            R_prev = R_new

        # Go forward by default
        robot_parts["wheel_left_joint"].setVelocity(vL)
        robot_parts["wheel_right_joint"].setVelocity(vR)

if mode == 'planner':

    
    
    
    if planning == "RRT":
        
        # Defining a class for each node on the path and
        class Node:
            def __init__(self, x, y, parent=None):
                self.point = (x, y)
                self.parent = parent
                self.cost = 0
                self.path_from_parent = []

        class RRT:

            # Initializing the RRT object
            def __init__(self, start, end, map, rand_area, expand_d, goal_prob=30, max_N=500):
                self.start = Node(start[0], start[1])
                self.end = Node(end[0], end[1])
                self.map = map
                self.rand_min = rand_area[0]
                self.rand_max = rand_area[1]
                self.expand_d = expand_d
                self.goal_prob = goal_prob
                self.max_N = max_N
                self.node_list = [self.start]

            def get_random_node(self):

                # Purpose: To obtain a random valid node

                # Occasionally set it to be the goal itself
                if np.random.randint(0, 100) > self.goal_prob:
                    # Find any point within the world bounds
                    row = np.random.randint(self.rand_min, 2*self.rand_max)
                    col = np.random.randint(self.rand_min, self.rand_max)

                    while self.map[row][col] == 1:
                        # reroll if it is in an object
                        row = np.random.randint(self.rand_min, self.rand_max)
                        col = np.random.randint(self.rand_min, self.rand_max)

                    rnd = [row, col]

                else:
                    rnd = [self.end.point[0], self.end.point[1]] # goal point

                return Node(rnd[0], rnd[1])

            def get_nearest_node_index(self, rand):
                
                # Purpose: Given a node, to find the closest node to it that's in the node list

                # Construct list of nearby distances
                dlist = [heuristic(node.point, rand.point) for node in self.node_list]

                # Return the minimum
                min_index = dlist.index(min(dlist))
                return min_index

            def check_path_valid(self, path):

                # Purpose: Check if path has nodes inside obstacles
                for coord in path:
                    if self.map[int(coord[0])][int(coord[1])] == 1:
                        return False

                return True

            def plan(self):

                # Purpose: Main method called to call all the other methods and generate an RRT path

                for i in range(self.max_N):

                    # Create a random node and get its nearest node
                    rand_node = self.get_random_node()
                    nearest_index = self.get_nearest_node_index(rand_node)
                    nearest_node = self.node_list[nearest_index]

                    # Steer towards another new node, this will be our test node to add or not
                    new_node = self.steer(nearest_node, rand_node, self.expand_d)

                    if self.check_collision(new_node, self.map):

                        # Find the nearest nodes to the new node and set a parent
                        near_inds = self.find_near_nodes(new_node)
                        node_with_shortest_d = self.choose_parent(new_node, near_inds)

                        # If found, add node to list and rewire the tree for optimal parents
                        if node_with_shortest_d:
                            print("Closest node is:", node_with_shortest_d.point)
                            self.node_list.append(new_node)
                            self.rewire(new_node, near_inds)

                # Find the goal node and generate a path that ends at that node, working backwards to the start point
                last_ind = self.search_goal_node()
                if last_ind is None:
                    return None
                path = self.generate_final_path(last_ind)

                # Return the final path
                return path

            def steer(self, from_node, to_node, extend_len=float("inf")):

                # Purpose: To take a step in the to_node's direction and returning a node in that direction

                # Initialize a new node at the from node's location, calculate d and theta to "to node"
                new_node = Node(from_node.point[0], from_node.point[1])
                d, theta = self.calc_d_and_theta(new_node, to_node)

                # Edge case handling
                if extend_len > d:
                    extend_len = d

                # Our new node should be at extend_d distance away from the first node
                new_node.point = (new_node.point[0] + int(extend_len * np.sin(theta)),
                                new_node.point[1] + int(extend_len * np.cos(theta)))
                
                # Set first node as parent of new node and return it
                new_node.parent = from_node
                return new_node

            def calc_d_and_theta(self, from_nd, to_nd):

                # Purpose: calculated d and theta from and to two nodes

                dx = to_nd.point[1] - from_nd.point[1]
                dy = to_nd.point[0] - from_nd.point[0]
                d = heuristic(from_nd.point, to_nd.point)
                theta = np.arctan2(dy, dx)

                # For this, we are specifically inputting nodes instead of point tuples
                return d, theta

            @staticmethod
            def check_collision(node_check, map_arr):

                # Purpose: To check if a node is inside an obstacle

                if map_arr[node_check.point[0]][node_check.point[1]] == 1:
                    return False
                else:
                    return True

            def find_near_nodes(self, new_node):

                # Purpose: to find and return the indices of the nearest nodes within the expand_d radius

                # number of nodes prospectively
                n_nodes = len(self.node_list) + 1

                # Bubble radius calculation
                r = self.expand_d * np.sqrt((np.log(n_nodes) / n_nodes))

                # Populating near inds array with the correct values and returning it
                near_inds = []
                for i, node in enumerate(self.node_list):
                    if heuristic(node.point, new_node.point) <= r:
                        near_inds.append(i)
                return near_inds

            def choose_parent(self, new_node, near_inds):
                
                # Purpose: To choose parent node for a new node
                if not near_inds:
                    return None

                # Evaluate cost values and calculate new costs to each of the near nodes relative to new node
                costs = []
                for i in near_inds:
                    near_node = self.node_list[i]
                    t_node = self.steer(near_node, new_node)
                    if t_node and self.check_collision(t_node, self.map):
                        costs.append(self.calc_new_cost(near_node, new_node))
                    else:
                        costs.append(float("inf"))
                min_cost = min(costs)

                # If cost is still infinity, no ideal parent can be assigned
                if min_cost == float("inf"):
                    print("No good path (min_cost is inf)")
                    return None

                # Find the index of the minimum cost node and steer towards it, set that as cost of new node
                min_ind = near_inds[costs.index(min_cost)]
                new_node = self.steer(self.node_list[min_ind], new_node)
                new_node.cost = min_cost

                return new_node

            def path_clear(self, node_1, node_2, map_array):

                # Purpose: Prototype method to find if two nodes are separated by or inside an obstacle

                line_xs = np.linspace(node_1.point[0], node_2.point[0], 10)
                line_ys = np.linspace(node_1.point[1], node_2.point[1], 10)

                for i in range(10):
                    if map_array[int(line_xs[i])][int(line_ys[i])] == 1:
                        return False

                return True


            def rewire(self, new_node, near_inds):

                # Purpose: Given a node, to redo costs and parent-assignments of tree after node is added

                for i in near_inds:

                    near_node = self.node_list[i]
                    edge_node = self.steer(new_node, near_node)
                    if not edge_node: continue
                    edge_node.cost = self.calc_new_cost(new_node, near_node)

                    no_collision = self.check_collision(edge_node, self.map)
                    valid_path = self.path_clear(edge_node, near_node, self.map)
                    improved_cost = near_node.cost > edge_node.cost

                    if no_collision and improved_cost and valid_path:
                        for node in self.node_list:
                            if node.parent == self.node_list[i]:
                                node.parent = edge_node
                        self.node_list[i] = edge_node
                        self.propagate_cost_to_leaves(self.node_list[i])


            def calc_new_cost(self, from_node, to_node):
                # Purpose: Update cost value using distance between nodes
                d = heuristic(from_node.point, to_node.point)
                return from_node.cost + d

            def propagate_cost_to_leaves(self, parent_node):
                # Purpose: Assign updated cost values recursively to parent node
                for node in self.node_list:
                    if node.parent == parent_node:
                        node.cost = self.calc_new_cost(parent_node, node)
                        self.propagate_cost_to_leaves(node)

            def search_goal_node(self):

                # Purpose: Look for goal node and return its index within the list of nodes

                dist_to_goal_list = [heuristic(n.point, self.end.point) for n in self.node_list]

                goal_inds = [i for i in range(len(dist_to_goal_list)) if dist_to_goal_list[i] <= self.expand_d]

                if len(goal_inds) == 0:
                    return None
                index = goal_inds[np.argmin([dist_to_goal_list[i] for i in goal_inds])]

                return index

            def generate_final_path(self, last_index):
                # Purpose: Work backwards from goal node to generate final path
                path = []
                node = self.node_list[last_index]
                while node is not None:
                    path.append(node)
                    node = node.parent
                return path[::-1]  # Reverses path

    elif planning == "Astar":

        def path_planner(map, start, end):
        '''
        :param map: A 2D numpy array of size 1400x700 representing the world's cspace with 0 as free space and 1 as obstacle
        :param start: A tuple of indices representing the start cell in the map
        :param end: A tuple of indices representing the end cell in the map
        :return: A list of tuples as a path from the given start to the given end in the given maze
        '''
        frontier = PriorityQueue()
        frontier.put(start, 0)
        cost = {start: 0}
        parent = {start: None}

        def heuristic(a, b):
            return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

        while not frontier.empty():
            current = frontier.get()
            if current == end:
                break

            space = [-25, 0, 25]
            for dx in space:
                for dy in space:
                    next = (current[0] + dx, current[1] + dy)
                    
                    if next[0] < 0 or next[0] >= map.shape[0] or next[1] < 0 or next[1] >= map.shape[1]:
                        continue
                    if map[next[0], next[1]] == 1:
                        continue
                    
                    # new cost should be sqrt(2) if diagonal
                    if dx != 0 and dy != 0:
                        new_cost = cost[current] + math.sqrt(2)
                    else:
                        new_cost = cost[current] + 1


                    if next not in cost or new_cost < cost[next]:
                        cost[next] = new_cost
                        priority = new_cost + heuristic(end, next)
                        frontier.put(next, priority)
                        parent[next] = current

        path = []
        current = end
        while current != start:
            path.append(current)
            current = parent[current]
        path.append(start)
        path.reverse()
        return path

    
if mode == 'path tester':
    path_loaded = np.load("path.npy")

    while robot.step(timestep) != -1:

        robot_parts["wheel_left_joint"].setVelocity(vL)
        robot_parts["wheel_right_joint"].setVelocity(vR)

        # Temporary odometry code
        left_pos = wheel_left_sensor.getValue()
        right_pos = wheel_right_sensor.getValue()
        L_new, R_new, pose_theta = compute_odometry(left_pos, right_pos)

        pose_x, pose_y, pose_theta = compute_odometry(left_pos, right_pos)

        gpose_x = gps.getValues()[0]
        gpose_y = gps.getValues()[1]

        pixel_x = res * 14 - int(pose_x * res)
        pixel_y = res * 7 - int(pose_y * res)

        for point in path_loaded:

            d, theta = calc_d_and_theta(point, (pixel_x, pixel_y))

            while d > 10:
                print("Theta between path points:", theta)
                print("Theta of pose:", pose_theta)
                if theta < pose_theta:
                    turn(robot, robot_parts, MAX_SPEED, "left")
                else:
                    turn(robot, robot_parts, MAX_SPEED, "right")

                d, theta = calc_d_and_theta(point, (pixel_x, pixel_y))

                robot_parts["wheel_left_joint"].setVelocity(vL)
                robot_parts["wheel_right_joint"].setVelocity(vR)

                left_pos = wheel_left_sensor.getValue()
                right_pos = wheel_right_sensor.getValue()
                L_new, R_new, pose_theta = compute_odometry(left_pos, right_pos)

                pose_x, pose_y, pose_theta = compute_odometry(left_pos, right_pos)

                gpose_x = gps.getValues()[0]
                gpose_y = gps.getValues()[1]

                pixel_x = res * 14 - int(pose_x * res)
                pixel_y = res * 7 - int(pose_y * res)


    # Part 2.1: Load map (map.npy) from disk and visualize it
    map = np.load(str(path_to_map))
    # plt.imshow(map)
    # plt.show()

    # Part 2.2: Compute an approximation of the “configuration space”
    robots_radius_meters = 0.35
    robot_radius_pixels = int(robots_radius_meters * res)
    kernel = np.ones((2 * robot_radius_pixels, 2 * robot_radius_pixels))
    conv_map = convolve2d(map, kernel, mode='same', boundary='fill', fillvalue=0)
    config_space_map = (conv_map > 0).astype(int)

    # Calling the path planner

    start_w = (-5, 0)  # (Pose_X, Pose_Y) in meters
    end_w = (0, -6)  # (Pose_X, Pose_Y) in meters

    # Convert the start_w and end_w from the webots coordinate frame into the map frame
    # start = world_to_map_coordinates(start_w) # (x, y) in 360x360 map
    # end = world_to_map_coordinates(end_w) # (x, y) in 360x360 map
    start_px = (int((14 - start_w[0]) * res), int((7 - start_w[1]) * res))
    end_px = (int((14 - end_w[0]) * res), int((7 - end_w[1]) * res))

    rho_5 = int(heuristic(start_px, end_px))

    path = path_planner(config_space_map, start_px, end_px)
    print(path)

    # rrt = RRT(start=start_px, end=end_px, map=config_space_map, rand_area=[0, 699], expand_d=rho_5, goal_prob=20, max_N=5000)
    # path = rrt.plan()
    #
    # distances = []
    # nodeslist_map = np.copy(config_space_map)
    # for node in rrt.node_list:
    #     nodeslist_map[node.point[0]][node.point[1]] = 1
    #     distances.append(heuristic(node.point, start_px))
    # nodeslist_map[end_px[0]][end_px[1]] = 1
    # plt.imshow(nodeslist_map)
    # plt.title("Map of added nodes to list")
    # plt.show()
    # print("MAX DISTANCE:", max(distances))

    if path is None:
        print("Cannot Find Path")
    else:
        print("Found path!")

        path_map = np.copy(config_space_map)
        for point in path:
            path_map[point[0]][point[1]] = 1
            print(point)
        # for node in path:
        #     path_map[node.point[0]][node.point[1]] = 1
        #     print(node.point)
        plt.imshow(path_map)
        plt.show()

        # Turning paths into waypoints and save on disk as path.npy and visualize it
        path = np.array(path)
        # waypoints = [(path[i].point[0]/res, path[i].point[1]/res) for i in range(len(path))]
        waypoints = [(path[i][0]/res, path[i][1]/res) for i in range(len(path))]
        np.save('waypoints.npy', waypoints)
        np.save('path.npy', path)



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
