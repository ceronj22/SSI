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

#cvae imports
from mushr_cvae_stripped import *

#joystick values
joy_states = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
hertz = 10


#average values for x and y particle filter position
x_pf_avg = []
y_pf_avg = []
#average values for z particle filter orientation
zorien_pf_avg = []



#just a set z
constant_z = torch.zeros(1).reshape(1, 1)
#if you want to alter that constant z...
constant_z[0][0] = 0.5
#constant_z[0][1] = 0.75

#our latent values
latent_z = torch.zeros(1).reshape(1, 1)
s = torch.zeros(3).reshape(1, 3)

const_throttle = 1
vel_scale = 0.5
turn_scale = 1 #0.34




#create a cVAE
net = cVAE()

app_folder = "/home/robot/catkin_ws/src/mushr/mushr_base/mushr_base/src/"
name = "mushr_cvae_weights"

checkpoint = torch.load(str(app_folder + name))
net.load_state_dict(checkpoint['model_state_dict'])





#pull particle filter values & append to average
def pf_callback(data):
    #convert to driver space
    x_pf = -data.pose.position.y
    y_pf = data.pose.position.x
    z_orien_pf = data.pose.orientation.z + 0.5
    
    x_pf_avg.append(x_pf) #current x pose - based on map
    y_pf_avg.append(y_pf) #current y pose
    zorien_pf_avg.append(z_orien_pf) #current z orientation


#scale the latent space for mapping
latent_scale = 3
#get joystick values
def joy_callback(data):
    joy_states[:] = data.axes[:]

    joy_states[3] = (joy_states[3] - 1) * -0.5

    latent_z[0][0] = joy_states[0] * latent_scale;# latent_z[0][1] = joy_states[1] * latent_scale


   
def send_command(pub_controls, cvae_output):
    #cVAE output interprets negative as driving forward, but publisher does opposite --- cVAE output interprets 0.5 as centered WA, but publisher says 0.0 is centered
    drive = AckermannDrive(speed = vel_scale * joy_states[3], steering_angle = (turn_scale * cvae_output[0] - 0.5))
    print(cvae_output)
    print(joy_states[3])

    #drive = AckermannDrive(steering_angle= turn_scale * joy_states[0], speed = vel_scale * (-joy_states[3] * joy_states[1]))
    pub_controls.publish(AckermannDriveStamped(drive=drive))



def timer_callback(data):
    #make sure we have data
    if len(x_pf_avg) > 0 and len(y_pf_avg) > 0 and len(zorien_pf_avg) > 0:
        #get state
        s[0][0] = sum(x_pf_avg) / len(x_pf_avg)
        s[0][1] = sum(y_pf_avg) / len(y_pf_avg)
        s[0][2] = sum(zorien_pf_avg) / len(zorien_pf_avg)
        
        cvae_output = cvae_to_action(net.P(latent_z, state_discretize(s)))
        send_command(pub_controls, cvae_output)
    #else:
    #    send_command(pub_controls, None)
    
    del x_pf_avg[:]
    del y_pf_avg[:]
    del zorien_pf_avg[:] 


    
def publisher():
    rospy.init_node("publisher", anonymous=True)
    rospy.Subscriber("/car/teleop/joy", sensor_msgs.msg.Joy, joy_callback)
    # particle filter
    rospy.Subscriber("/car/particle_filter/inferred_pose", geometry_msgs.msg.PoseStamped, pf_callback)
    
    #call the timer callback every 1/hertz (1/10) second
    rospy.Timer(rospy.Duration(1.0 / hertz), timer_callback)
    rospy.spin()

    
if __name__ == "__main__": 
    print("Running CVAE Assisted Teleop!")

    control_topic = rospy.get_param("~control_topic", "/car/mux/ackermann_cmd_mux/input/teleop") 
    pub_controls = rospy.Publisher(control_topic, AckermannDriveStamped, queue_size=1) 
 
    init_pose_topic = rospy.get_param("~init_pose_topic", "/initialpose") 
    pub_init_pose = rospy.Publisher(init_pose_topic, PoseWithCovarianceStamped, queue_size=1) 
 
    publisher()
    print("Quit CVAE Assisted Teleop")
