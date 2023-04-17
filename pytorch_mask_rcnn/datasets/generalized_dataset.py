import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from torchvision import transforms

# 数据集定义的父类
class GeneralizedDataset:
    """
    Main class for Generalized Dataset.
    """
    
    def __init__(self, max_workers=2, verbose=False):
        self.max_workers = max_workers
        self.verbose = verbose
            
    # 输入序号，返回RGB图像的张量形式，和该图像所有标注框的信息字典
    def __getitem__(self, i):
        img_id = self.ids[i]
        image = self.get_image(img_id)
        image = transforms.ToTensor()(image)
        target = self.get_target(img_id) if self.train else {}
        return image, target   
    
    def __len__(self):
        return len(self.ids)
    
    # 检查图像，保证它们的宽高比符合要求
    def check_dataset(self, checked_id_file):
        """
        use multithreads to accelerate the process.
        check the dataset to avoid some problems listed in method `_check`.
        """
        
        # 如果当前路径已经检查过了，则直接返回跳过
        if os.path.exists(checked_id_file):
            info = [line.strip().split(", ") for line in open(checked_id_file)]
            self.ids, self.aspect_ratios = zip(*info)
            return
        
        since = time.time()
        print("Checking the dataset...")
        
        # 多线程配置
        executor = ThreadPoolExecutor(max_workers=self.max_workers) # 创建一个线程池执行器
        seqs = torch.arange(len(self)).chunk(self.max_workers)  # 将数据集索引范围进行划分，将所有数据集的索引分为self.max_workers份
        tasks = [executor.submit(self._check, seq.tolist()) for seq in seqs]    # 使用线程池执行器进行检查，每个线程任务检查范围为seq.tolist()，所有任务保存在tasks列表中

        outs = []
        # 执行tasks中的_check，并将结果保存在outs列表中
        for future in as_completed(tasks):
            outs.extend(future.result())
        # 创建一个将字符串转换为整数的lambda函数
        if not hasattr(self, "id_compare_fn"):
            self.id_compare_fn = lambda x: int(x)
        # 将outs列表按照id值排序（每个元素的第一个值x[0]）
        outs.sort(key=lambda x: self.id_compare_fn(x[0]))
        
        # 将检查过的图像的id和宽高比（img_id, aspect_ratio）添加到checked_id_file中
        with open(checked_id_file, "w") as f:
            for img_id, aspect_ratio in outs:
                f.write("{}, {:.4f}\n".format(img_id, aspect_ratio))
         
        # 将 checked_id_file 文件的每一行作为一个字符串读取
        info = [line.strip().split(", ") for line in open(checked_id_file)]
        # 使用 zip(*info) 方法将图像ID和宽高比分别解压缩成两个元组，并将它们分别赋值给 self.ids 和 self.aspect_ratios 变量
        self.ids, self.aspect_ratios = zip(*info)
        print("checked id file: {}".format(checked_id_file))
        print("{} samples are OK; {:.1f} seconds".format(len(self), time.time() - since))
        
    # 输入：seq（数据集序列）
    # 返回：out（数据列表，获取每个图像的id和对应的宽高比，并作为一个元组(img_id, aspect_ratio)添加到输出列表out中）
    def _check(self, seq):
        out = []
        for i in seq:
            # 读取第i个对象的数据
            img_id = self.ids[i]
            target = self.get_target(img_id)
            boxes = target["boxes"]
            labels = target["labels"]
            masks = target["masks"]

            # 检查数据是否正确
            try:
                assert len(boxes) > 0, "{}: len(boxes) = 0".format(i)
                assert len(boxes) == len(labels), "{}: len(boxes) != len(labels)".format(i)
                assert len(boxes) == len(masks), "{}: len(boxes) != len(masks)".format(i)

                # 如果正确，获取每个图像的id和对应的宽高比，并作为一个元组(img_id, aspect_ratio)添加到输出列表out中
                out.append((img_id, self._aspect_ratios[i]))
            except AssertionError as e:
                if self.verbose:
                    print(img_id, e)
        return out

                    