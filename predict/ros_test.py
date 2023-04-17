#!/home/huffie/anaconda3/envs/densefusion18/bin/python

import cv2
from PIL import Image

# ROS
from ros.image_subscriber import ImageSubscriber, array_publisher
import rospy

if __name__ == '__main__':
    rospy.init_node('image_subscriber')
    ic = ImageSubscriber()
    while not rospy.is_shutdown():
        image = ic.get_image()
        depth = ic.get_depth()

        cv2.namedWindow("image")
        cv2.imshow("image", image)
        # cv2.imshow("depth", depth)
        cv2.waitKey(1)
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("Shutting down")
    cv2.destroyAllWindows()