import torch
import pytorch_mask_rcnn as pmr
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2

use_cuda = True
dataset = "coco"
ckpt_path = "./check_points/maskrcnn_coco-50.pth"
# data_dir = "./dateset"

device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
if device.type == "cuda":
    pmr.get_gpu_prop(show=True)
print("\ndevice: {}".format(device))

image = Image.open("/home/huffie/Documents/PoseEstimation/MaskDenseFusion/dateset/train2017/0019.jpg")
# image = Image.open("/home/huffie/Documents/PoseEstimation/DenseFusion/datasets/linemod/LINEMOD/data/01/rgb/0123.png")
image = image.convert("RGB")
image = transforms.ToTensor()(image)

classes = {1: 'ammeter', 2: 'coffeebox', 3: 'realsensebox', 4: 'sucker'}
model = pmr.maskrcnn_resnet50(True, max(classes) + 1).to(device)
model.eval()
model.head.score_thresh = 0.3

if ckpt_path:
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    # print(checkpoint["eval_info"])
    del checkpoint

for p in model.parameters():
    p.requires_grad_(False)

image = image.to(device)
# target = {k: v.to(device) for k, v in target.items()}

with torch.no_grad():
    result = model(image)

pmr.show(image, result, classes, "./images/output.jpg")

# print(result)

labels = result["labels"].tolist()
masks = result["masks"].tolist()
scores = result["scores"].tolist()

label_num = len(labels)
i = 0
for label in labels:
    if scores[i] > 0.7:
        print(label)
        # print(masks[i])
        # 列表转灰度图
        gray_img = (np.array(masks[i]) * 255).astype(np.uint8)
        # 二值化图像
        threshold_value = 128
        max_value = 255
        binary_img = cv2.threshold(gray_img, threshold_value, max_value, cv2.THRESH_BINARY)[1]

        cv2.imshow("binary image", binary_img)
        cv2.waitKey(0)
    i += 1

    
