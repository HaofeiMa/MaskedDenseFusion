import os
from PIL import Image

import torch
from .generalized_dataset import GeneralizedDataset
       
        
class COCODataset(GeneralizedDataset):
    def __init__(self, data_dir, split, train=False):
        super().__init__()
        from pycocotools.coco import COCO
        
        self.data_dir = data_dir
        self.split = split
        self.train = train
        
        ann_file = os.path.join(data_dir, "annotations/instances_{}.json".format(split))    # 读取注释文件
        self.coco = COCO(ann_file)
        self.ids = [str(k) for k in self.coco.imgs]
        
        # 分类值必须从1开始，因为0在模型中代表背景
        self.classes = {k+1: v["name"] for k, v in self.coco.cats.items()}
        
        # 加载检查文件
        checked_id_file = os.path.join(data_dir, "checked_{}.txt".format(split))
        # 进行训练前，如果数据未检查，则进行检查操作
        if train:
            if not os.path.exists(checked_id_file):
                self._aspect_ratios = [v["width"] / v["height"] for v in self.coco.imgs.values()]
            self.check_dataset(checked_id_file)
    
    # 输入：img_id（图像序号）
    # 返回：RGB图像
    def get_image(self, img_id):
        img_id = int(img_id)
        img_info = self.coco.imgs[img_id]
        image = Image.open(os.path.join(self.data_dir, "{}".format(self.split), img_info["file_name"]))
        return image.convert("RGB")
    
    @staticmethod
    def convert_to_xyxy(boxes): # box format: (xmin, ymin, w, h)
        x, y, w, h = boxes.T
        return torch.stack((x, y, x + w, y + h), dim=1) # new_box format: (xmin, ymin, xmax, ymax)
        
    # 输入：img_id（图像序号）
    # 返回：target（字典，包含该图像的id，所有标注框的边界框，标签和mask掩码数组）
    def get_target(self, img_id):
        img_id = int(img_id)
        ann_ids = self.coco.getAnnIds(img_id)   # 获取该图片的所有标注框的ID（列表）
        anns = self.coco.loadAnns(ann_ids)  # 加载该图像的所有标注框（列表）
        boxes = []
        labels = []
        masks = []

        # 提取标注框的信息
        if len(anns) > 0:
            for ann in anns:
                boxes.append(ann['bbox'])
                labels.append(ann["category_id"])
                mask = self.coco.annToMask(ann) # 返回一个二维numpy数组
                mask = torch.tensor(mask, dtype=torch.uint8)    # 转换为张量形式
                masks.append(mask)  # 并添加到mask列表中

            boxes = torch.tensor(boxes, dtype=torch.float32)
            boxes = self.convert_to_xyxy(boxes) # 将标注框由(xmin, ymin, w, h)变为(xmin, ymin, xmax, ymax)格式
            labels = torch.tensor(labels)
            masks = torch.stack(masks)

        # 创建一个词典，包含标注框的所有信息
        target = dict(image_id=torch.tensor([img_id]), boxes=boxes, labels=labels, masks=masks)
        return target
    
    