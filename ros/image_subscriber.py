#!/home/huffie/anaconda3/envs/densefusion18/bin/python

import rospy
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

bridge = CvBridge()
rgb_image = None
depth_image = None

#################################### ROS图像订阅 ##########################################

# define the callback function for the color image
def rgb_callback(data):
    global rgb_image
    rgb_image = bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")


# define the callback function for the depth image
def depth_callback(data):
    global depth_image
    depth_image = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
    # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)


#################################### ROS发布函数 ##########################################
def array_publisher(array_data):
    # 初始化ROS节点
    rospy.init_node('pred_publisher', anonymous=True)

    # 创建发布者 "/pred_points" 消息类型为 "Float32MultiArray"
    pub = rospy.Publisher('/pred_points', Float32MultiArray, queue_size=10)

    # 创建 Float32MultiArray 消息
    array_msg = Float32MultiArray()

    # 将 numpy 数组放入消息中
    array_msg.data = array_data.flatten().tolist()

    # 以10HZ 发布消息
    rate = rospy.Rate(10) 
    while not rospy.is_shutdown():
        pub.publish(array_msg)
        rate.sleep()





if __name__ == '__main__':
    rospy.init_node('realsense_display', anonymous=True)
    rospy.Subscriber('/camera/depth/image_rect_raw', Image, depth_callback)
    rospy.Subscriber('/camera/color/image_raw', Image, rgb_callback)

    while not rospy.is_shutdown():
        if depth_image is not None and rgb_image is not None:
            cv2.imshow("RGB Image", rgb_image)
            cv2.imshow("Depth Image", depth_image)
            print(depth_image)
            cv2.waitKey(1)