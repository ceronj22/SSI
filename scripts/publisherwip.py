\#!/usr/bin/env python
import importlib
from threading import Thread

import actionlib
import genpy.message
import numpy as np
import rospy
import rosservice
import rostopic
from rospy import ROSException
from rosservice import ROSServiceException
from stf_msgs.msg import Float64, Bool
import sensor msgs.msg
import geometry_msgs.msg

from cvae.py import *

button_states = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
joy_states = [0.0,0.0,0.0,0.0]
motor_speed = []
wheel_angle = []
hertz = 10
x_pf = 0
y_pf = 0
constant_z =(0,0)
throttle = 0.0

def joy_callback(data):
    button_states[:] = data.buttons[:]
    joy_states[:] = data.axes

def motor_speed(data):
    motor_speed.append(data.data)

def wheel_angle_callback(data):
    wheel_angle.append(data.data)

def joy_to_latent():
    latent_z = (joy_states[0], latent_y = joy_states[1])

def pf_callback(data):
    x_pf = data.pose.position.x
    y_pf = data.pose.position.y
    z_orien_pf = data.pose.orientation.z
    s = (x_pf,y_pf,z_orien_pf)

def cvae_action(z, state):
    cvae_action = motor_speed, wheel_angle
    return(cvae_action)

def send_command()
    drive = AckermanDrive(steering_angle= throttle * (cvae_action[0]), speed=throttle * (cvae_action[1]))   

def conv_driver_space():
    #mutate pf space to driver space
    s[0] = -s[0]
    s[2] =  s[2] + 1/2

def timer_callback():
    rospy.init_node("listener", anonymous=True)
    rospy.Timer(rospy.Duration(1.0 / hertz, timer_callback)
    rospy.Subscriber("/car/teleop/joy", sensor_msgs.Joy, joy_callback)
    rospy.Subscriber("/car/particle_filter/inferred_pose", geometry_msgs.msg.PoseStamped, pf_callback)
    conv_driver_space()
    joy_to_latent_map(constant_z)
    cvae_action(constant_latent, s)
    send_command()
    rospy.spin()
    
if __name__ == "__main__": 
    rospy.init_node("publisher") 
    print("Running CVAE Assisted Teleop!)
    timer_callback()
    print("Quit CVAE Assisted Teleop")
 
    

