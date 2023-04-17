################### 初始化 #####################
import torch
import pytorch_mask_rcnn as pmr
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image

use_cuda = True
dataset = "coco"
ckpt_path = "./check_points/20230223_Phone_4Obj_Aug_Coco.pth"
data_dir = "./dateset"

device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
if device.type == "cuda":
    pmr.get_gpu_prop(show=True)
print("\ndevice: {}".format(device))

################### 定义函数 ####################
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

    return mapped_bbox


################### 加载模型 ####################
classes = {1: 'ammeter', 2: 'coffeebox', 3: 'realsensebox', 4: 'sucker'}
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


###################### 读取图片 ################################
# image = Image.open("/home/huffie/Pictures/mask_rcnn_dataset/0011.jpg")
image = Image.open("/home/huffie/Documents/ImageProcessing/labelme2coco/images/val2017/02_4OBJ_20230222_083543_aug_1.png")
image_np = np.array(image)
image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

image = image.convert("RGB")
image = transforms.ToTensor()(image)
image = image.to(device)


######################## 利用网络获得结果 ###########################
with torch.no_grad():
    result = model(image)

print(result)

############################ 可视化 ###############################
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

# 判断boxes大小是否超出640x480
isOutrange = False
for i, box in enumerate(boxes):
    x1, y1, x2, y2 = [int(x) for x in box]
    if (x2 - x1 >= 640) or (y2 - y1 >= 480):
        isOutrange = True

# 调整图像大小，同时将边界框和轮廓做映射
if isOutrange:
    overlay_masks = combine_masks(image_cv2, masks) # 合并所有masks图像
    image_cv2_resize = cv2.resize(image_cv2, (640, 480))    # resize为640x480
    overlay_masks_resize = cv2.resize(overlay_masks, (640, 480))
    new_boxes = resize_boxes(image_cv2, boxes, (640, 480))   # 重新映射boxes位置
    # for i in range(len(boxes)):
    #     mapped_bbox, mapped_contours = resize_annotations(image_cv2, boxes[i], masks[i], (640, 480))
    #     new_boxes.append(mapped_bbox)
    #     new_masks.appen(mapped_contours)
    cv2.imshow("new_masks", overlay_masks_resize)
    img_resize = overlay_instances_resize(image_cv2_resize, overlay_masks_resize, new_boxes, labels)
    cv2.imshow("img_res/ize",img_resize)
else:
    # 调用可视化函数，显示结果
    img = overlay_instances(image_cv2, masks, boxes, labels)
    cv2.imshow("img",img)

key = cv2.waitKey(0)