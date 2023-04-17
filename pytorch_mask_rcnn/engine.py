import sys
import time

import torch

from .utils import Meter, TextArea
try:
    from .datasets import CocoEvaluator, prepare_for_coco
except:
    pass

def train_one_epoch(model, optimizer, data_loader, device, epoch, args):
    # 加载参数
    for p in optimizer.param_groups:
        p["lr"] = args.lr_epoch

    iters = len(data_loader) if args.iters < 0 else args.iters

    t_m = Meter("total")
    m_m = Meter("model")
    b_m = Meter("backward")
    model.train()
    A = time.time()

    # 开始训练
    for i, (image, target) in enumerate(data_loader):   # 迭代读取数据
        T = time.time()
        num_iters = epoch * len(data_loader) + i
        if num_iters <= args.warmup_iters:  # 当当前迭代的编号 num_iters 小于等于预设的 warm-up 迭代数 args.warmup_iters 时
            r = num_iters / args.warmup_iters
            for j, p in enumerate(optimizer.param_groups):
                p["lr"] = r * args.lr_epoch # 将学习率逐渐增加到预设的学习率 args.lr_epoch。
                   
        image = image.to(device)
        target = {k: v.to(device) for k, v in target.items()}   # 将数据和标注放到 GPU 上
        S = time.time()
        
        losses = model(image, target)   # 前向计算模型，并计算模型的损失
        total_loss = sum(losses.values())
        m_m.update(time.time() - S)
            
        S = time.time()
        total_loss.backward()   # 反向传播，计算模型中所有参数的梯度
        b_m.update(time.time() - S)
        
        optimizer.step()    # 使用优化器更新模型参数，并将梯度清零
        optimizer.zero_grad()

        if num_iters % args.print_freq == 0:    # 每 args.print_freq 个迭代打印一次损失。损失的值来自 losses 字典
            print("{}\t".format(num_iters), "\t".join("{:.3f}".format(l.item()) for l in losses.values()))

        t_m.update(time.time() - T)
        if i >= iters - 1:  # 达到 iters 迭代数时退出循环
            break
           
    A = time.time() - A
    print("iter: {:.1f}, total: {:.1f}, model: {:.1f}, backward: {:.1f}".format(1000*A/iters,1000*t_m.avg,1000*m_m.avg,1000*b_m.avg))
    return A / iters
            

def evaluate(model, data_loader, device, args, generate=True):
    iter_eval = None
    if generate:
        iter_eval = generate_results(model, data_loader, device, args)

    dataset = data_loader #
    iou_types = ["bbox", "segm"]
    coco_evaluator = CocoEvaluator(dataset.coco, iou_types) # 使用给定的数据集和IOU类型（即边框和分割掩码）创建一个 CocoEvaluator 对象

    results = torch.load(args.results, map_location="cpu")  

    S = time.time()
    coco_evaluator.accumulate(results)  # 使用 accumulate() 函数将评估结果累积到评估器中
    print("accumulate: {:.1f}s".format(time.time() - S))

    # collect outputs of buildin function print
    temp = sys.stdout
    sys.stdout = TextArea()

    coco_evaluator.summarize()

    output = sys.stdout
    sys.stdout = temp
        
    return output, iter_eval
    
    
# generate results file   
@torch.no_grad()   
def generate_results(model, data_loader, device, args):
    iters = len(data_loader) if args.iters < 0 else args.iters
        
    t_m = Meter("total")
    m_m = Meter("model")
    coco_results = []
    model.eval()
    A = time.time()
    for i, (image, target) in enumerate(data_loader):
        T = time.time()
        
        image = image.to(device)
        target = {k: v.to(device) for k, v in target.items()}   # 将图像移动到指定设备上

        S = time.time()
        #torch.cuda.synchronize()
        output = model(image)   # 通过模型进行预测，得到预测结果
        m_m.update(time.time() - S)
        
        prediction = {target["image_id"].item(): {k: v.cpu() for k, v in output.items()}}
        coco_results.extend(prepare_for_coco(prediction))   # 将预测结果保存在coco_result中

        t_m.update(time.time() - T)
        if i >= iters - 1:
            break
     
    A = time.time() - A 
    print("iter: {:.1f}, total: {:.1f}, model: {:.1f}".format(1000*A/iters,1000*t_m.avg,1000*m_m.avg))
    torch.save(coco_results, args.results)
        
    return A / iters
    

