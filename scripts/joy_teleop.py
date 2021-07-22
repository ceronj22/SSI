#!/usr/bin/env python
import importlib
from threading import Thread

from tf.transformations import quaternion_from_euler 
import actionlib
import genpy.message
import numpy as np
import rospy
import rosservice
import rostopic
from rospy import ROSException
from rosservice import ROSServiceException
from std_msgs.msg import Float64, Bool
import sensor_msgs.msg
import geometry_msgs.msg
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped 
from geometry_msgs.msg import ( 
    Point, 
    Pose, 
    PoseWithCovariance, 
    PoseWithCovarianceStamped, 
    Quaternion, 
) 

#from mushr_cvae.py import *

button_states = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
joy_states = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
motor_speed = []
wheel_angle = []
hertz = 10
x_pf = 0
y_pf = 0
constant_z =(0,0)
const_throttle = 1
vel_scale = 0.5
turn_scale = 0.34

def joy_callback(data):
    button_states[:] = data.buttons[:]
    joy_states[:] = data.axes
    

def motor_speed(data):
    motor_speed.append(data.data)

def wheel_angle_callback(data):
    wheel_angle.append(data.data)

def joy_to_latent():
    latent_z = (joy_states[0], joy_states[1])

def pf_callback(data):
    x_pf = data.pose.position.x
    y_pf = data.pose.position.y
    z_orien_pf = data.pose.orientation.z
    s = (x_pf,y_pf,z_orien_pf)

   

#def cvae_action(z, state):
#    cvae_action = motor_speed, wheel_angle
#    return(cvae_action)

def send_command(pub_controls):
    #drive = AckermannDrive(steering_angle= cvae_action[1], speed= const_throttle * (cvae_action[0]))  
    drive = AckermannDrive(steering_angle= turn_scale * joy_states[0], speed = vel_scale * (-joy_states[3] * joy_states[1]))
    pub_controls.publish(AckermannDriveStamped(drive=drive))  
def conv_driver_space():
    #mutate pf space to driver space
    s[0] = -s[0]
    s[2] =  s[2] + 1/2


def timer_callback(data):
    #conv_driver_space()
    #joy_to_latent_map(constant_z)
    #cvae_action(constant_latent, s)
    send_command(pub_controls)
    
def publisher():
    rospy.init_node("publisher", anonymous=True)
    rospy.Subscriber("/car/teleop/joy", sensor_msgs.msg.Joy, joy_callback)
    #rospy.Subscriber("/car/particle_filter/inferred_pose", geometry_msgs.msg.PoseStamped, pf_callback)
    rospy.Subscriber("/car/car_pose", geometry_msgs.msg.PoseStamped, pf_callback)
    rospy.Timer(rospy.Duration(1.0 / hertz), timer_callback)
    print(joy_states)
    rospy.spin()

    
if __name__ == "__main__": 
    print("Running CVAE Assisted Teleop!")

    control_topic = rospy.get_param("~control_topic", "/car/mux/ackermann_cmd_mux/input/teleop") 
    pub_controls = rospy.Publisher(control_topic, AckermannDriveStamped, queue_size=1) 
 
    init_pose_topic = rospy.get_param("~init_pose_topic", "/initialpose") 
    pub_init_pose = rospy.Publisher(init_pose_topic, PoseWithCovarianceStamped, queue_size=1) 
 

 
    publisher()
    print("Quit CVAE Assisted Teleop")
 
    

