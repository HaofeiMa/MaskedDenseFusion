# MaskedDenseFusion
A pose estimation method combining instance segmentation

# 1. MaskedDenseFusion

I have implemented a 6D pose estimation method for objects based on RGBD information, which combines instance segmentation.
(1) Obtain object masks through instance segmentation;
(2) Crop to obtain RGB and depth information of the area where the object is located;
(3) Extract RGB and depth information features using convolutional networks, and then perform feature fusion;
(4) Calculate the loss by combining the feature vectors with the features of the model;
(5) Regress the translation and rotation of the object relative to the camera coordinate system.

# 2. Demo and Performance

https://user-images.githubusercontent.com/49356049/232419809-fe899a21-65b7-4430-8c8e-c46175635143.mp4

# 3. Acknowledgements

This project is extended base on
1. [https://github.com/matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)
2. [https://github.com/j96w/DenseFusion](https://github.com/j96w/DenseFusion)
