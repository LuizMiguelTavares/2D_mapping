import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tb3_ra_interfaces.msg import GridMap
from tf_transformations import euler_from_quaternion
import math
import numpy as np

class TurtlebotControl(Node):

    def __init__(self):
        super().__init__("turtlebot3_control")

        self.angle_adjustment = None            # ângulo que deve ser ajustado
        self.wall_dist        = 0.4             # Distância mínima que pode chegar de um objeto
        self.pose             = None            # Guarda a posição e velocidade do robô
        self.graus            = None            # Possui a orientação do robô
        self.msg              = Twist()         # movimenta o robô
        self.ang_min_range    = None            # ângulo com o menor range
        self.min_range        = None
        self.rotation         = 90              # ângulo de rotação
        
        # Switch loops
        self.case_cl          = 1               # Case para o "control loop"
        self.case_ffo         = 1               # Case para "find first obstacle" 
        self.case_fw          = 1               # Case para "follow_wall"

        self.erro = 0
        self.p = 0.6
        self.d = 0.85
        self.i = 0.15

        self.x_target = None
        self.y_target = None

        self.yaw = None

        self.laser_const = 1
        
        self.rotation_gain = 0.1
        self.forward_gain  = 0.1
        self.safe_margin   = 0.01

        self.rotation_flag = None

        self.dist_270 = None

        # Publishing frame rate
        self.frames_sec = 5

        # Creating motion control publisher
        self.motion_publisher_ = self.create_publisher(Twist, "cmd_vel", 10) # Publishes in the cmd_vel topic from turtlebot3

        self.pose_subscription = self.create_subscription(Odometry, "odom", self.Turtlebot3_pose, 10)

        self.laser_subscriber_ = self.create_subscription(LaserScan, "scan", self.laser_reading, 10)

        self.pose_subscription = self.create_subscription(GridMap, "grid_map", self.to_go, 10)
        
        self.timer_ = self.create_timer(1/self.frames_sec, self.control_loop) # Run control_loop 30 times per sec
        
        self.get_logger().info("Turtlebot3 Node has been started!!")
    
    def Turtlebot3_pose(self, msg): # OK
        self.pose = msg
        self.position = msg.pose.pose.position
        self.orientation = msg.pose.pose.orientation
        self.orientation = [self.orientation.x, self.orientation.y, self.orientation.z, self.orientation.w]
        (self.roll, self.pitch, self.yaw) = euler_from_quaternion(self.orientation)

    def laser_reading(self, msg): # OK
        self.ranges = msg.ranges

        if self.ranges is None:
            return

        self.const = 3.0
        min_dist   = 0.3
        '''
        Laser sentido anti-horário - pesando pro sentido horário (42 graus para cada lado)
        '''
        if msg.ranges[2*self.laser_const] == float("inf"):
            msg.ranges[2*self.laser_const] = self.const

        if msg.ranges[2*self.laser_const]<=min_dist:
            self.dist_003 = 1/(msg.ranges[2*self.laser_const] * 0.90)
        else:
            self.dist_003 = 0.0
        ###
        if msg.ranges[13*self.laser_const] == float("inf"):
            msg.ranges[13*self.laser_const] = self.const
        
        if msg.ranges[13*self.laser_const]<=min_dist:
            self.dist_014 = 1/(msg.ranges[13*self.laser_const] * 0.935)
        else:
            self.dist_014 = 0.0
        ###

        if msg.ranges[27*self.laser_const] == float("inf"):
            msg.ranges[27*self.laser_const] = self.const
        
        if msg.ranges[27*self.laser_const]<=min_dist:
            self.dist_028 = 1/(msg.ranges[27*self.laser_const] * 0.97)
        else:
            self.dist_028 = 0.0
        ###

        if msg.ranges[41*self.laser_const] == float("inf"):
            msg.ranges[41*self.laser_const] = self.const

        if msg.ranges[41*self.laser_const]<=min_dist:
            self.dist_042 = 1/(msg.ranges[41*self.laser_const])
        else:
            self.dist_042 = 0.0

        # Ponto lateral
        if msg.ranges[89*self.laser_const] == float("inf"):
            msg.ranges[89*self.laser_const] = self.const

        if msg.ranges[89*self.laser_const]<=min_dist:
            self.dist_090 = 1/(msg.ranges[89*self.laser_const]*0.85) # Aumento de 15% no peso (Quanto menor o valor maior o ganho)
        else:
            self.dist_090 = 0.0

        '''
        Laser sentido horário - pesando pro sentido anti-horário (42 graus para cada lado)
        '''
        if msg.ranges[356*self.laser_const] == float("inf"):
            msg.ranges[356*self.laser_const] = self.const
        
        if msg.ranges[356*self.laser_const]<=min_dist:
            self.dist_357 = 1/(msg.ranges[356*self.laser_const] * 0.90)
        else:
            self.dist_357 = 0.0
        ###
        if msg.ranges[345*self.laser_const] == float("inf"):
            msg.ranges[345*self.laser_const] = self.const

        if msg.ranges[345*self.laser_const]<=min_dist:
            self.dist_346 = 1/(msg.ranges[345*self.laser_const] * 0.94)
        else:
            self.dist_346 = 0.0
        ###
        if msg.ranges[331*self.laser_const] == float("inf"):
            msg.ranges[331*self.laser_const] = self.const

        if msg.ranges[331*self.laser_const]<=min_dist:
            self.dist_332 = 1/(msg.ranges[331*self.laser_const] * 0.97)
        else:
            self.dist_332 = 0.0
        ###
        if msg.ranges[317*self.laser_const] == float("inf"):
            msg.ranges[317*self.laser_const] = self.const
        
        if msg.ranges[317*self.laser_const]<=min_dist:
            self.dist_318 = 1/(msg.ranges[317*self.laser_const])
        else:
            self.dist_318 = 0.0

        # Ponto lateral
        if msg.ranges[269*self.laser_const] == float("inf"):
            msg.ranges[269*self.laser_const] = self.const
        
        if msg.ranges[269*self.laser_const]<=min_dist:
            self.dist_270 = 1/(msg.ranges[269*self.laser_const] * 0.85) # Aumento de 15% no peso (Quanto menor o valor maior o ganho)
        else:
            self.dist_270 = 0.0

        self.rot =(self.dist_003 + self.dist_014 + self.dist_028 + self.dist_042 + self.dist_090)/5 - (self.dist_357 + self.dist_346 + self.dist_332 + self.dist_318 + self.dist_270)/5
    
    def to_go(self, msg):
        if msg is None:
            return

        self.x_target = msg.x_to_go
        self.y_target = msg.y_to_go

    def control_loop(self):
        # X = -2
        # Y = -0.5
        if self.x_target is None:
            return
        
        aux = 0
        x_position = self.position.x
        y_position = self.position.y

        x_target = self.x_target
        y_target = self.y_target

        x = x_target - x_position
        y = y_target - y_position

        #self.get_logger().info('Rot gain: ' + str(round(self.rot, 2)))
        
        #self.get_logger().info(str(math.atan(-1/1)))
        #self.get_logger().info(str(y))
        if y >=0 and x>=0:
            obj = math.atan(y/x)
        elif y>=0 and x<0:
            obj = np.pi + math.atan(y/x)
        elif y<0 and x>=0:
            obj = math.atan(y/x)
        else:
            obj = -np.pi + math.atan(y/x)

        #self.get_logger().info('atan(' + str(round(y, 2)) + '/' + str(round(x, 2)) + ') = ' + str(round(obj, 2)))

        if self.yaw is None:
            return

        if obj>=0:
            f_obj = True
        else:
            f_obj = False

        if self.yaw>=0:
            f_yaw = True
        else:
            f_yaw = False

        ########################

        if f_obj==f_yaw:
            erro = obj - self.yaw 
        else:
            if f_obj:
                aux = -np.pi+obj
            else:
                aux = np.pi+obj
            
            if f_yaw:
                if self.yaw>=aux:
                    erro = self.yaw - obj 
                else:
                    erro = obj - self.yaw 
            else:
                if self.yaw>=aux:
                    erro = obj - self.yaw 
                else:
                    erro = self.yaw - obj 

        #self.get_logger().info(str(round(obj, 2)) + ' - ' + str(round(self.yaw,2)) + '= ' + str(round(erro, 5)))
        #self.get_logger().info('Erro: ' + str(round(erro, 5)))
        #self.get_logger().info('X position: '+ str(round(x_position, 4)) + ',  y position: ' + str(round(y_position,4)))
        
        position_error = np.sqrt((x_position-x_target)**2 + (y_position-y_target)**2)
        if abs(erro)>0.01:
            self.msg.angular.z = 0.7*erro # Positivo = anti-horário
        else:
            self.msg.angular.z = 0.0
        
        if self.msg.angular.z > 0.4:
            self.msg.angular.z = 0.4

        self.msg.angular.z  = self.msg.angular.z - (self.rot*0.5)
        
        #self.get_logger().info(str(position_error))

        if abs(erro)<0.261799:
            if position_error>0.25:
                self.msg.linear.x = 0.3
            else:
                self.msg.linear.x = 0.0
                self.msg.angular.z = 0.0
        else:
            self.msg.linear.x = 0.06
        
        self.motion_publisher_.publish(self.msg)

def main(args=None):
    rclpy.init(args=args)
    node = TurtlebotControl()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()