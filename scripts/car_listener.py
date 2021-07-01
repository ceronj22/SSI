#!/usr/bin/env python

#7.1.2021

import rospy
from std_msgs.msg import Float64

import csv


#Header to dictate values in each column
cols = ["Velocity"]

#2D list to store all the values in
csv_data = []

#how many times it will grab information in one second
hertz = 10



#get the data from the publisher and print to console
def callback(data):
    #rospy.loginfo(rospy.get_caller_id() + 'I heard %s', data.data)
    
    #add information to the data list
    csv_data.append([data.data])
    
    #print the data to console console
    rospy.loginfo('I heard and stored %s, csv_data len: %s', csv_data[-1], len(csv_data))
    

def car_listener():
    
    #pauses the loop for the duration of 1/hertz seconds - problem: pulling info way faster than displaying
    #rospy.sleep(1.0/hertz)
    
    rospy.init_node("car_listener", anonymous=True)
    
    rospy.Subscriber("/car/vesc/commands/motor/speed", Float64, callback)
    
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()





#takes collected data and writes it to the csv file in the same directory
def write_to_csv():
    #open the file - cleaner than having to close seperately
    with open('test.csv', 'w+') as file:
        #create a csv writer
        writer = csv.writer(file)
        
        #add the header to the csv file for convenience
        writer.writerow(cols)
        
        #write all rows to that csv file
        writer.writerows(csv_data)
        print("Data saved to csv!")


if __name__ == '__main__':
    

    print("Running main!")
    car_listener()
    print("Quit out of the listener.")

    write_to_csv()

