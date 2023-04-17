# MaskRCNN
import torch
import pytorch_mask_rcnn as pmr
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
# ROS
import rospy
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image as rosImage
from cv_bridge import CvBridge, CvBridgeError

'''
—————————————————————————————————— 全局变量 ——————————————————————————————————————————————
'''

##################### MaskRCNN #####################
use_cuda = True
dataset = "coco"
ckpt_path = "./check_points/20230223_Phone_4Obj_Aug_Coco.pth"
data_dir = "./dateset"
device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
if device.type == "cuda":
    pmr.get_gpu_prop(show=True)
print("\ndevice: {}".format(device))

####################### ROS #######################
bridge = CvBridge()
rgb_image = None
depth_image = None

'''
—————————————————————————————————— 函数定义 ——————————————————————————————————————————————
'''

################### MaskRCNN函数定义 ####################
# 获得标记文本
def create_text_labels(classes, scores, class_names):
    labels = None
    if classes is not None:
        if class_names is not None and len(class_names) > 0:
            labels = [class_names[i] for i in classes]
        else:
            labels = [str(i) for i in classes]
    if scores is not None:
        if labels is None:
            labels = ["{:.0f}%".format(s * 100) for s in scores]
        else:
            labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]
    return labels

# 可视化函数：已知boxes和masks，绘制轮廓与矩形框
def overlay_instances(img, masks, boxes, labels):
    np.random.seed(66)
    COLORS = np.random.uniform(0, 255, size=(80, 3))

    # 将所有实例的掩膜叠加并转换为三通道的二值化掩膜
    overlay_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for i, mask in enumerate(masks):
        color = COLORS[labels[i]]
        overlay_mask = np.maximum(overlay_mask, (np.array(mask) > 0.5).astype(np.uint8) * 255)
    
    # 找到所有轮廓
    contours, hierarchy = cv2.findContours(overlay_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 在图像上绘制所有轮廓
    cv2.drawContours(img, contours, -1, (255, 255, 255), 2)

    # 在图像上绘制每个实例的类别标签和置信度得分
    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]
        x1, y1, x2, y2 = [int(x) for x in box]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{classes[labels[i]]}: {scores[i]*100:.2f}%", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return img

# 可视化函数：获取轮廓叠加态
def combine_masks(img, masks):
    # 将所有实例的掩膜叠加并转换为三通道的二值化掩膜
    overlay_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for i, mask in enumerate(masks):
        overlay_mask = np.maximum(overlay_mask, (np.array(mask) > 0.5).astype(np.uint8) * 255)
    return overlay_mask


def overlay_instances_resize(img, overlay_mask, boxes, labels):
    np.random.seed(66)
    COLORS = np.random.uniform(0, 255, size=(80, 3))

    # 在图像上绘制所有轮廓
    contours, hierarchy = cv2.findContours(overlay_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (255, 255, 255), 2)

    # 在图像上绘制每个实例的类别标签和置信度得分
    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]
        x1, y1, x2, y2 = [int(x) for x in box]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{classes[labels[i]]}: {scores[i]*100:.2f}%", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return img


    return contours
# 降采样后的边界框和轮廓
def resize_boxes(image, bbox_list, target_size):
    """
    将图像、边界框和轮廓缩放为指定大小
    :param image: 原图像
    :param bbox_list: 边界框列表，每个元素为 [x1, y1, x2, y2] 格式的列表
    :param contour_list: 轮廓列表，每个元素为包含点坐标的列表
    :param target_size: 目标大小，格式为 (width, height)
    :return: 缩放后的图像，边界框列表和轮廓列表
    """
    h, w = image.shape[:2]
    new_w, new_h = target_size

    # 计算宽高比例
    w_ratio = new_w / w
    h_ratio = new_h / h

    # 缩放边界框
    mapped_bbox = []
    for bbox in bbox_list:
        x1, y1, x2, y2 = bbox
        x1 = int(x1 * w_ratio)
        y1 = int(y1 * h_ratio)
        x2 = int(x2 * w_ratio)
        y2 = int(y2 * h_ratio)
        mapped_bbox.append([x1, y1, x2, y2])


    # resized_contour = []
    # for point in contour_list:
    #     print(point)
    #     x, y = point
    #     x = int(x * w_ratio)
    #     y = int(y * h_ratio)
    #     resized_contour.append([x, y])
    # mapped_mask = np.zeros((480, 640), dtype=np.uint8)
    # cv2.drawContours(mapped_mask, resized_contour, -1, 255, -1)


    return mapped_bbox

##################### ROS函数定义 ##############################
# ROS图像订阅
# define the callback function for the color image
def rgb_callback(data):
    global rgb_image
    rgb_image = bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")


# define the callback function for the depth image
def depth_callback(data):
    global depth_image
    depth_image = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
    # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)


# ROS发布函数
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

'''
—————————————————————————————————— 主函数 ——————————————————————————————————————————————
'''

if __name__ == '__main__':
    ################### 加载MaskRCNN模型 ####################
    # classes = {1: 'ammeter', 2: 'coffeebox', 3: 'realsensebox', 4: 'sucker'}
    classes = {1: 'ammeter', 2: 'realsensebox', 3: 'coffeebox', 4: 'sucker'}
    model = pmr.maskrcnn_resnet50(True, max(classes) + 1).to(device)
    model.eval()
    model.head.score_thresh = 0.3

    if ckpt_path:
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        print(checkpoint["eval_info"])
        del checkpoint

    for p in model.parameters():
        p.requires_grad_(False)


    ###################### ROS初始化 #########################
    rospy.init_node('realsense_display', anonymous=True)
    rospy.Subscriber('/camera/depth/image_rect_raw', rosImage, depth_callback)
    rospy.Subscriber('/camera/color/image_raw', rosImage, rgb_callback)



    ####################### 主循环 ##########################
    while not rospy.is_shutdown():
        # 加载Realsense图像
        if depth_image is None or rgb_image is None:
            continue
        image_cv2 = rgb_image
        image = Image.fromarray(cv2.cvtColor(image_cv2,cv2.COLOR_BGR2RGB))
        image = image.convert("RGB")
        image = transforms.ToTensor()(image)
        image = image.to(device)
        # target = {k: v.to(device) for k, v in target.items()}

        # 利用MaskRCNN网络获得结果
        with torch.no_grad():
            result = model(image)
        
        # 判断结果
        if isinstance(image, torch.Tensor) and image.dim() == 3:
            images = [image]
        if isinstance(result, dict):
            targets = result    # [result]

        # 进行标注
        class_names = classes
        for i in range(len(images)):
            if targets is not None:
                # 获取数据
                # list indices must be integers or slices, not str
                # print(targets)
                boxes = targets["boxes"].tolist() if "boxes" in targets else None
                scores = targets["scores"].tolist() if "scores" in targets else None
                labels = targets["labels"].tolist() if "labels" in targets else None
                masks = targets["masks"].tolist() if "masks" in targets else None

        # 图像降采样
        overlay_masks = combine_masks(image_cv2, masks) # 合并所有masks图像
        image_cv2_resize = cv2.resize(image_cv2, (640, 480))    # resize为640x480
        overlay_masks_resize = cv2.resize(overlay_masks, (640, 480))
        new_boxes = resize_boxes(image_cv2, boxes, (640, 480))   # 重新映射boxes位置
        cv2.imshow("new_masks", overlay_masks_resize)
        img_resize = overlay_instances_resize(image_cv2_resize, overlay_masks_resize, new_boxes, labels)
        cv2.imshow("img_res/ize",img_resize)


        key = cv2.waitKey(1)
        #print(key)
        if key  == ord('q'):  #判断是哪一个键按下
            break

    cv2.destroyAllWindows()
