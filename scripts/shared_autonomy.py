#Nick Cerone
#7.28.21

#!/usr/bin/env python

import rospy
from geometry_msgs.msg import (
    Point,
    Pose,
    PoseWithCovariance,
    PoseWithCovarianceStamped,
    Quaternion,
)
from tf.transformations import quaternion_from_euler


#how many times per second the timer callback should be run
hertz = 10


#takes a goal position in x,y,theta form and publishes to /move_base_simple/goal
def send_goal_pose(pub_goal_pose, goal):
    #convert goal data out of string form
    goal_data = goal.split(",") #goal is of form "x,y,theta"
    assert len(goal_data) == 3

    #set x, y, and theta based on information from goal_data
    x, y, theta = float(goal_data[0]), float(goal_data[1]), float(goal_data[2])
    #get a publishable quaternion orientation
    q = Quaternion(*quaternion_from_euler(0, 0, theta))
    point = Point(x=x, y=y)
    #create a PoseWithCovariance object to publish containing point x,y and orientation q(theta)
    pose = PoseWithCovariance(pose=Pose(position=point, orientation=q))
    
    #publish to the pub_goal_pose topic at /move_base_simple/goal
    pub_goal_pose.publish(PoseWithCovarianceStamped(pose=pose))

    

#runs every 1/hertz seconds
def timer_callback(data):
    goal = "1,1,0.5" #goal is of form "x,y,theta"
    send_goal_pose(pub_goal_pose, goal)
  
    
#loops through rostopic subscribers
def publisher():
    rospy.init_node("publisher", anonymous=True)
    
    #rospy.Subscriber("/car/teleop/joy", sensor_msgs.msg.Joy, joy_callback)
    #rospy.Subscriber("/car/particle_filter/inferred_pose", geometry_msgs.msg.PoseStamped, pf_callback)
    
    #call the timer callback
    rospy.Timer(rospy.Duration(1.0 / hertz), timer_callback)
    
    #loop
    rospy.spin()
    

    
if __name__ == "__main__":
    print("Running Shared Autonomy Goal Pose Publisher!")
    
    goal_pose_topic = rospy.get_param("~goal_pose_topic", "/move_base_simple/goal")
    pub_goal_pose = rospy.Publisher(goal_pose_topic, PoseWithCovarianceStamped, queue_size=1)

    publisher()
    
    print("Ended Shared Autonomy Goal Pose Publisher.")
