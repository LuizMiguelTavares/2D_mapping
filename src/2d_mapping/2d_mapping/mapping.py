import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan 
from nav_msgs.msg import OccupancyGrid

from tf_transformations import euler_from_quaternion
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

class MapVisualizer(Node):

    def __init__(self):
        super().__init__('map_visualizer')

        self.subscription = self.create_subscription(
            OccupancyGrid,
            'map_data',
            self.visualize_map,
            10
        )

    def visualize_map(self, msg):
        if len(msg.data) == 0:
            self.get_logger().info('Received empty map data. Skipping visualization.')
            return
        self.get_logger().info('Received!')
        occupancy_data = np.array(msg.data)

        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution

        # Reshape the occupancy data array to match the grid dimensions
        occupancy_map = occupancy_data.reshape((height, width))

        # Create a figure and axis for the plot
        
        #plt.switch_backend('TkAgg')
        # Plot the occupancy grid
        #plt.imshow(occupancy_map, cmap='gray', origin='lower')
        #plt.show()

class MapCreator(Node):

    def __init__(self):
        super().__init__("map_creator")

        self.teste = True
        self.laser_const     = 1
        self.frames_sec      = 10
        self.resolution      = 0.05
        self.grid_resolution = 0.005

        self.pose = None # Inicializando a pose
        self.ranges = None
        self.map = None
        
        self.group = ReentrantCallbackGroup()
        self.pose_subscriber = self.create_subscription(Odometry, "odom", self.Turtlebot3_pose, 40, callback_group=self.group)
        self.laser_subscriber = self.create_subscription(LaserScan, "scan", self.laser_reading, 10, callback_group=self.group)
        self.GridMap_publisher = self.create_publisher(OccupancyGrid, 'map_data', 10)
        self.timer_ = self.create_timer(1/self.frames_sec, self.loop, callback_group=self.group)
        #self.get_logger().info("Mapping Node has been started!!")

    def Turtlebot3_pose(self, msg): # OK
        self.pose = msg
        self.orientation = msg.pose.pose.orientation
        self.position = msg.pose.pose.position
        self.orientation = [self.orientation.x, self.orientation.y, self.orientation.z, self.orientation.w]
        (self.roll, self.pitch, self.yaw) = euler_from_quaternion(self.orientation)
        self.pose_timestamp = self.pose.header.stamp.sec+self.pose.header.stamp.nanosec*10e-9

        #self.pose_timestamp = time.perf_counter()

        #self.get_logger().info(str(self.pose_timestamp))
        #self.get_logger().info(str(time.perf_counter()))
        #self.get_logger().info('Pose')

    def laser_reading(self, msg): # OK
        # Sentido anti-horário
        self.ranges = msg.ranges
        self.laser_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec*10e-9
        #self.laser_timestamp = time.perf_counter()
        #self.get_logger().info(str(self.laser_timestamp))
        #self.get_logger().info('Laser')

    def grid_point(self, n):
        return round(n/self.grid_resolution)*self.grid_resolution

    def calculate_cartesian_coordinates(self, robot_ranges, robot_x_position, robot_y_position, robot_yaw):
        
        x = []
        y = []
        weight = []
        degree = []
        ranges = []

        x_ps = []
        y_ps = []
        
        yaw = robot_yaw
        x_p = robot_x_position
        y_p = robot_y_position
        ranges_laser = robot_ranges

        R1 = np.matrix([[1, 0, x_p] , [0, 1, y_p], [0, 0, 1]])
        R2 = np.matrix([[math.cos(yaw), -math.sin(yaw), 0] , [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
        
        M = np.dot(R1,R2)

        for i in range(len(ranges_laser)):
            
            if ranges_laser[i] == float("inf"):
                    ranges_laser[i] = 4

            for j in range(int(ranges_laser[i]/self.resolution)):
                degree.append(i/self.laser_const)
                ranges.append(ranges_laser[i])
                const = round((j+1)/int(ranges_laser[i]/self.resolution), 2)
                    
                if ranges_laser[i] < 4:

                    P  = np.matrix([[const * ranges_laser[i] * math.cos(math.radians(i/self.laser_const))],[const * ranges_laser[i] * math.sin(math.radians(i/self.laser_const))],[1]])
                    
                    x.append(self.grid_point((np.dot(R1, np.dot(R2,P)))[0,0]))
                    y.append(self.grid_point((np.dot(R1, np.dot(R2,P)))[1,0]))

                    if const == 1.0:
                        x_ps.append(P[0,0])
                        y_ps.append(P[1,0])

                        weight.append(int(1))
                    else:
                        weight.append(int(-1))
                else:
                    P  = np.matrix([[const * 3.5 * math.cos(math.radians(i/self.laser_const))],[const * 3.5 * math.sin(math.radians(i/self.laser_const))],[1]])
                    
                    x.append(self.grid_point((np.dot(M,P))[0,0]))
                    y.append(self.grid_point((np.dot(M,P))[1,0]))

                    weight.append(int(-1))
        
        return x, y, weight, degree, ranges

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def publish_map(self, x, y, weigths):

        min_x = np.min(x)
        min_y = np.min(y)
        max_x = np.max(x)
        max_y = np.max(y)

        x_new = np.array(x) + np.abs(min_x)
        y_new = np.array(y) + np.abs(min_y)

        # print(len(x_new))
        # print(len(x))
        print(int(np.max(x_new)/self.resolution))
        print(int(np.max(y_new)/self.resolution))
        print(np.min(x_new))
        print(np.min(y_new))

        #local_width = math.ceil((max_x - min_x)/self.resolution)
        #local_height = math.ceil((max_y - min_y)/self.resolution)
        local_width = int(np.max(x_new)/self.resolution)+1
        local_height = int(np.max(y_new)/self.resolution)+1

        local_aux_map = np.zeros((local_width, local_height))
        print(local_aux_map.shape)

        # for i in range(len(x_new)):
        #     local_aux_map[int(x_new[i]/self.resolution), int(y_new[i]/self.resolution)] = weigths[i]
        
        for i in range(len(x_new)):
            x_index = int(x_new[i] / self.resolution)
            y_index = int(y_new[i] / self.resolution)

            # Check if indices are within bounds
            if x_index >= 0 and x_index < local_width and y_index >= 0 and y_index < local_height:
                local_aux_map[x_index, y_index] = weigths[i]
            else:
                print(x_index, y_index)

        #self.map = None

        if self.map is None:

            self.map = local_aux_map

            width  = local_width
            height = local_height

            self.min_x_map = min_x
            self.min_y_map = min_y
            self.max_x_map = max_x
            self.max_y_map = max_y
        
        else:
            lower_x = np.min([min_x, self.min_x_map])
            higher_x = np.max([max_x, self.max_x_map])
            lower_y = np.min([min_y, self.min_y_map])
            higher_y = np.max([max_y, self.max_y_map])

            width = math.ceil((higher_x - lower_x)/self.resolution)+1
            height = math.ceil((higher_y - lower_y)/self.resolution)+1

            aux_map = np.zeros((width, height))

            relative_y_min = math.ceil(abs(self.min_y_map - min_y)/self.resolution)
            relative_x_min = math.ceil(abs(self.min_x_map - min_x)/self.resolution)

            if self.min_x_map < min_x:
                if self.min_y_map < min_y:

                    aux_map[0:self.map.shape[0], 0:self.map.shape[1]] = self.map
                    aux_map[relative_x_min:(relative_x_min+local_aux_map.shape[0]), relative_y_min:(local_aux_map.shape[1]+relative_y_min)] = aux_map[relative_x_min:(relative_x_min+local_aux_map.shape[0]), relative_y_min:(local_aux_map.shape[1]+relative_y_min)] + local_aux_map

                else:
                    aux_map[0:self.map.shape[0], relative_y_min:(self.map.shape[1]+relative_y_min)] = self.map
                    aux_map[relative_x_min:(relative_x_min+local_aux_map.shape[0]), 0:local_aux_map.shape[1]] = aux_map[relative_x_min:(relative_x_min+local_aux_map.shape[0]), 0:local_aux_map.shape[1]] + local_aux_map
            else:
                if self.min_y_map < min_y:
                    aux_map[relative_x_min:(relative_x_min+self.map.shape[0]), 0:self.map.shape[1]] = self.map
                    aux_map[0:local_aux_map.shape[0], relative_y_min:(local_aux_map.shape[1]+relative_y_min)] = aux_map[0:local_aux_map.shape[0], relative_y_min:(local_aux_map.shape[1]+relative_y_min)] + local_aux_map

                else:
                    aux_map[relative_x_min:(relative_x_min+self.map.shape[0]), relative_y_min:(self.map.shape[1]+relative_y_min)] = self.map
                    aux_map[0:local_aux_map.shape[0], 0:local_aux_map.shape[1]] = aux_map[0:local_aux_map.shape[0], 0:local_aux_map.shape[1]] + local_aux_map

            self.map = aux_map
        print('oi')
        #plt.scatter(x, y, c=weigths)
        print(np.min((self.sigmoid(self.map)*100).astype(np.int8)))
        print(np.max((self.sigmoid(self.map)*100).astype(np.int8)))
        plt.imshow(((self.sigmoid(self.map)-0.5)*100).astype(np.int8), cmap='gray', origin='lower', vmin=-100,vmax=100)
        plt.show()
        map_1d = np.array(self.sigmoid(self.map.reshape(-1))*100).astype(np.int8)

        msg = OccupancyGrid()
        msg.header.frame_id = 'map'
        msg.info.map_load_time.sec = 0
        msg.info.resolution = self.resolution  # Cell size in meters
        msg.info.width = width  # Number of cells in the x-direction
        msg.info.height = height  # Number of cells in the y-direction
        msg.info.origin.position.x = 0.0
        msg.info.origin.position.y = 0.0
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.x = 0.0
        msg.info.origin.orientation.y = 0.0
        msg.info.origin.orientation.z = 0.0
        msg.info.origin.orientation.w = 1.0
        msg.data = map_1d.tolist()

        self.GridMap_publisher.publish(msg)
        


    def loop(self):
        if self.ranges is None or self.pose is None:
            self.get_logger().info('Unable to identify the pose and/or laser data of the robot')
            return None

        #self.get_logger().info(str(abs(self.pose_timestamp - self.laser_timestamp)))

        if abs(self.pose_timestamp - self.laser_timestamp) > 0.1:
            return None
        
        x, y, weight, degree, ranges = self.calculate_cartesian_coordinates(robot_ranges=np.array(self.ranges),  robot_x_position=self.position.x, robot_y_position=self.position.y, robot_yaw=self.yaw)
        self.publish_map(x, y, weight)

def main(args=None):
    rclpy.init(args=args)
    try:
        Map_Creator_node = MapCreator()
        Map_Visualizaer_node = MapVisualizer()

        executor = SingleThreadedExecutor() #MultiThreadedExecutor(num_threads=4)
        executor.add_node(node=Map_Creator_node)
        executor.add_node(node=Map_Visualizaer_node)

        try: 
            executor.spin()
        finally:
            executor.shutdown()
            Map_Creator_node.destroy_node()
            Map_Visualizaer_node.destroy_node()

    finally:
        rclpy.shutdown()

if __name__ == "__main__":
    main()