#!/usr/bin/env python

#7.5.2021

import rospy
from std_msgs.msg import Float64, Bool #motor speed, wheel angle, pushbutton
import sensor_msgs.msg #joystick
import geometry_msgs.msg


import csv



#2D list to store all the values in - start off with a header
csv_data = [["Velocity", "Wheel Angle", "Push Button Pressed", "X Pose", "Y Pose"]]



#state of all of the controller buttons
#               sq, x, O,tri,L1,R1,L2,R2,sh,st,ls,rs,ps,mid plate 
button_states = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]



#how many times you want to pull info per second
hertz = 10











#array to store motor speed vals before timer callback runs
motor_speed_avg = []

#get the data from the publisher
def motor_speed_callback(data):

    #only store the info if the square button is pressed:
    if button_states[4] == 1:
        motor_speed_avg.append(data.data)






#array to store wheel angle vals before timer callback runs
wheel_angle_avg = []

#get the data from the publisher
def wheel_angle_callback(data):
    
    #only store the info if the square button is pressed:
    if button_states[4] == 1:
        wheel_angle_avg.append(data.data)



#is the push button pressed?
global push_button_state
push_button_state = False

#get the data from the publisher
def push_button_callback(data):
    global push_button_state
    
    #only store the info if the square button is pressed (trivial here, but uniform):
    if button_states[4] == 1:
        push_button_state = data.data





#average values for x position and y position
x_pose_avg = []
y_pose_avg = []

def pose_callback(data): # docs.ros.org/en/noetic/api/geometry_msgs/html/msg/PoseStamped.html

    #only store the info if the square button is pressed:
    if button_states[4] == 1:
        x_pose_avg.append(data.pose.position.x) #current x pose - universal for the duration of teleop
        y_pose_avg.append(data.pose.position.y) #current y pose
        #print('({}, {})'.format(x_pose_avg[-1], y_pose_avg[-1]))







#get button vals from the teleop/joy topic
def joy_callback(data):
    #set the global button states array to the data we just recieved
    button_states[:] = data.buttons[:]
    
    #rospy.loginfo('The button states are: %s', button_states)








#run every 1/10 of a second and add vals to csv_data
def timer_callback(data):
    #temp list to store all the values we just got
    to_append = []
    
    #don't try to append if there are no values
    if len(motor_speed_avg) > 0 and len(wheel_angle_avg) > 0 and len(x_pose_avg) > 0 and len(y_pose_avg) > 0:
        #add average motor speed to the to_append list
        to_append.append(sum(motor_speed_avg) / len(motor_speed_avg))
        
        #add average motor speed to the to_append list
        to_append.append(sum(wheel_angle_avg) / len(wheel_angle_avg))
        
        #add push button state to the to_append list
        to_append.append(push_button_state)
        
        #add x and y pose to the to_append list
        to_append.append(sum(x_pose_avg) / len(x_pose_avg))
        to_append.append(sum(y_pose_avg) / len(y_pose_avg))
        
        
        
        #append all the vals to csv data
        csv_data.append(to_append)
        
        
        rospy.loginfo('%s appended | csv_data len: %s', csv_data[-1], len(csv_data)-1)
        
        
        #clear average arrays
        del motor_speed_avg[:]
        del wheel_angle_avg[:]
        del x_pose_avg[:]
        del y_pose_avg[:]






#calls all the callbacks - single stream, but not necessarily in order
def car_listener():
    
    rospy.init_node("car_listener", anonymous=True)
    
    #call the timer callback
    rospy.Timer(rospy.Duration(1.0 / hertz), timer_callback)
    
    
    #call the joystick callback
    rospy.Subscriber("/car/teleop/joy", sensor_msgs.msg.Joy, joy_callback)
    
    
    #call the motor speed callback
    rospy.Subscriber("/car/vesc/commands/motor/speed", Float64, motor_speed_callback)
    
    #call the wheel angle callback
    rospy.Subscriber("/car/vesc/commands/servo/position", Float64, wheel_angle_callback)
    
    #call the push button callback
    rospy.Subscriber("/car/push_button_state", Bool, push_button_callback)

    #call the pose callback
    rospy.Subscriber("/car/car_pose", geometry_msgs.msg.PoseStamped, pose_callback)

    
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()







#takes collected data and writes it to the csv file in the same directory
def write_to_csv():
    #open the file - cleaner than having to close seperately
    with open('test.csv', 'w+') as file:
        #create a csv writer
        writer = csv.writer(file)
        
        print('csv_data: {}'.format(csv_data))
        
        #write all rows to that csv file
        writer.writerows(csv_data)
        print("Data saved to csv!")



#run the callbacks until the program is quit, at which point write to csv
if __name__ == '__main__':
    
    print("Running main!")
    car_listener()
    print("Quit out of the listener.")
    
    write_to_csv()
