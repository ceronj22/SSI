#!/usr/bin/env python

#7.5.2021

import rospy
from std_msgs.msg import Float64
import sensor_msgs.msg

import csv


#Header to dictate values in each column
cols = ["Velocity", "Wheel Angle"]

#2D list to store all the values in
csv_data = []



#state of all of the controller buttons
#               sq, x, O,tri,L1,R1,L2,R2,sh,st,ls,rs,ps,mid plate 
button_states = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]



#boolean flipped by timer callback - only pull data when true
global get_data 
get_data = False

#how many times you want to pull info per second
hertz = 10




motor_speed_avg = []

#get the data from the publisher and print to console
def motor_speed_callback(data):
    global get_data 
    
    #rospy.loginfo(rospy.get_caller_id() + 'I heard %s', data.data)
        
    #print(button_states)
    
    #only store the info if the square button is pressed:
    if button_states[4] == 1:
        if get_data:
            #add information to the data list
            csv_data.append([sum(motor_speed_avg)/len(motor_speed_avg)])
            
            #print the data to console console
            rospy.loginfo('I heard and stored %s, csv_data len: %s', csv_data[-1], len(csv_data))
            
            #make sure you wont get data until another 1/hertz seconds passes
            #wheel angle is called after, so you only want to set get_data to salse once that passes
            #get_data = False
            
            del motor_speed_avg[:]
        else:
            motor_speed_avg.append(data.data)


wheel_angle_avg = []

#get the data from the publisher and print to console
def wheel_angle_callback(data):
    global get_data
    
    #rospy.loginfo(rospy.get_caller_id() + 'I heard %s', data.data)
    
    #only store the info if the square button is pressed:
    if button_states[4] == 1:
        if get_data:
            #add information to the latest list within data
            csv_data[-1].append(sum(wheel_angle_avg)/len(wheel_angle_avg))
            
            #print the data to console console
            rospy.loginfo('I heard and stored %s, csv_data len: %s', csv_data[-1], len(csv_data))
            
            #make sure you wont get data until another 1/hertz seconds passes
            get_data = False
            
            del wheel_angle_avg[:]
        else:
            #print("{} was just added to wheel_angle_avg".format(data.data))
            wheel_angle_avg.append(data.data)




#get button vals from the teleop/joy topic
def joy_callback(data):
    #set the global button states array to the data we just recieved
    button_states[:] = data.buttons[:]
    
    #rospy.loginfo('The button states are: %s', button_states)



#run every 1/10 of a second and set get_data to true (so the motor speed can be added to data)
def timer_callback(data):
    global get_data
    
    #make sure there's stuff in the array before trying to append to the 2D csv_data array
    if len(motor_speed_avg) > 0 and len(wheel_angle_avg) > 0:
        get_data = True




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

    
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()





#takes collected data and writes it to the csv file in the same directory
def write_to_csv():
    #open the file - cleaner than having to close seperately
    with open('test.csv', 'w+') as file:
        #create a csv writer
        writer = csv.writer(file)
        
        #add the header to the csv file for clarity
        writer.writerow(cols)
        
        #write all rows to that csv file
        writer.writerows(csv_data)
        print("Data saved to csv!")



#run the callbacks until the program is quit, at which point write to csv
if __name__ == '__main__':
    
    print("Running main!")
    car_listener()
    print("Quit out of the listener.")
    
    write_to_csv()
