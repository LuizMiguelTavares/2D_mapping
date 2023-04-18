import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf_transformations import euler_from_quaternion
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class MapCreator(Node):

    def __init__(self):
        super().__init__("map_creator")

        self.teste = True
        self.laser_const = 1
        self.frames_sec  = 0.2
        self.resolution = 0.1

        self.pose = None # Inicializando a pose
        self.ranges = None
        
        self.pose_subscriber = self.create_subscription(Odometry, "odom", self.Turtlebot3_pose, 40)
        self.laser_subscriber = self.create_subscription(LaserScan, "scan", self.laser_reading, 10)
        self.timer_ = self.create_timer(1/self.frames_sec, self.polar_to_cartesian_global)
        self.get_logger().info("Mapping Node has been started!!")

    def Turtlebot3_pose(self, msg): # OK
        self.pose = msg
        self.orientation = msg.pose.pose.orientation
        self.position = msg.pose.pose.position
        self.orientation = [self.orientation.x, self.orientation.y, self.orientation.z, self.orientation.w]
        (self.roll, self.pitch, self.yaw) = euler_from_quaternion(self.orientation)
        self.get_logger().info('pose')

    def laser_reading(self, msg): # OK
        # Sentido anti-horário
        self.ranges = msg.ranges
        self.get_logger().info('laser')

    def polar_to_cartesian_global(self):

        x = []
        y = []
        weight = []
        degree = []
        ranges = []

        x_ps = []
        y_ps = []

        if self.ranges == None or self.pose == None:
            return None, None, None, None, None
        
        yaw     = self.yaw
        x_p     = self.position.x
        y_p     = self.position.y
        ranges_laser  = np.array(self.ranges)

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
                    
                    x.append((np.dot(R1, np.dot(R2,P)))[0,0])
                    y.append((np.dot(R1, np.dot(R2,P)))[1,0])

                    if const == 1.0:
                        x_ps.append(P[0,0])
                        y_ps.append(P[1,0])

                        weight.append(int(0))
                    else:
                        weight.append(int(100))
                else:
                    P  = np.matrix([[const * 3.5 * math.cos(math.radians(i/self.laser_const))],[const * 3.5 * math.sin(math.radians(i/self.laser_const))],[1]])
                    
                    x.append((np.dot(M,P))[0,0])
                    y.append((np.dot(M,P))[1,0])

                    weight.append(int(100))
        
        
        t = pd.DataFrame(data={'x':x_ps, 'y':y_ps})
        t.to_csv('teste')
        d = {'x':x, 'y':y, 'w':weight}

        if self.teste:
            self.get_logger().info('teste 1')
            self.dt = pd.DataFrame(data=d)
            self.teste = False
        else:
            
            self.get_logger().info('oi')
            dt2 = pd.DataFrame(data=d)
            self.dt = pd.concat([self.dt,dt2], ignore_index=True)

        plt.scatter(self.dt['x'], self.dt['y'], s=1, c = self.dt['w'])
        plt.scatter(x_p , y_p, color='green', s=30)

        plt.show()
        self.get_logger().info('yaw = ' + str(round(yaw, 3)) + ', x = ' + str(round(x_p, 2)) + ', y = ' + str(round(y_p, 2)))
        return x, y, weight, degree, ranges

def main(args=None):
    rclpy.init(args=args)
    node = MapCreator()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()