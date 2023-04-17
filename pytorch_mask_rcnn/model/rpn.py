import torch
import torch.nn.functional as F
from torch import nn

from .box_ops import BoxCoder, box_iou, process_box, nms
from .utils import Matcher, BalancedPositiveNegativeSampler

# RPN区域建议网络的头部模块
# 输入：输入特征通道数，每个位置默认框的数量
# 返回：对应分类得分，边界框预测
class RPNHead(nn.Module):
    def __init__(self, in_channels, num_anchors):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)    # 一个3x3的卷积层
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, 1)    # 一个输出num_anchors个通道的卷积层，用于分类得分
        self.bbox_pred = nn.Conv2d(in_channels, 4 * num_anchors, 1) # 一个输出4xnum_anchors通道的卷积层，用于边界框预测
        
        for l in self.children():
            nn.init.normal_(l.weight, std=0.01) # 权重初始化为均值0、标准差0.01的正态分布
            nn.init.constant_(l.bias, 0)        # 偏置初始化为0
    
    # 接受输入特征图，进行前向计算，输出每个位置默认框的分类得分和边界框预测
    def forward(self, x):
        x = F.relu(self.conv(x))
        logits = self.cls_logits(x)
        bbox_reg = self.bbox_pred(x)
        return logits, bbox_reg
    
# RPN区域建议网络
# 输入：
    # anchor_generator: 一个生成候选框的对象。
    # head: 一个接收特征表示作为输入的神经网络。
    # fg_iou_thresh: 正样本的 IoU 阈值。
    # bg_iou_thresh: 负样本的 IoU 阈值。
    # num_samples: 样本数量。
    # positive_fraction: 正样本占比。
    # reg_weights: 用于回归的权重。
    # pre_nms_top_n: 非极大抑制前要保留的建议数量。
    # post_nms_top_n: 非极大抑制后要保留的建议数量。
    # nms_thresh: 非极大抑制的阈值。
class RegionProposalNetwork(nn.Module):
    def __init__(self, anchor_generator, head, 
                 fg_iou_thresh, bg_iou_thresh,
                 num_samples, positive_fraction,
                 reg_weights,
                 pre_nms_top_n, post_nms_top_n, nms_thresh):
        super().__init__()
        
        self.anchor_generator = anchor_generator
        self.head = head
        
        self.proposal_matcher = Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=True)
        self.fg_bg_sampler = BalancedPositiveNegativeSampler(num_samples, positive_fraction)
        self.box_coder = BoxCoder(reg_weights)
        
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = 1
                
    # 根据分类结果和回归偏移量生成候选框
    # 输入：Anchor框，是否包含有目标的分类分数，预测的边界框偏移量，图像大小
    # 输出：经过NMS和处理后的建议框
    def create_proposal(self, anchor, objectness, pred_bbox_delta, image_shape):
        # 训练时使用训练参数、测试时使用测试参数
        if self.training:
            pre_nms_top_n = self._pre_nms_top_n['training']
            post_nms_top_n = self._post_nms_top_n['training']
        else:
            pre_nms_top_n = self._pre_nms_top_n['testing']
            post_nms_top_n = self._post_nms_top_n['testing']
            
        pre_nms_top_n = min(objectness.shape[0], pre_nms_top_n)
        top_n_idx = objectness.topk(pre_nms_top_n)[1]   # 选择前k个置信度最高的建议框
        score = objectness[top_n_idx]   # 获取这些建议框的得分
        proposal = self.box_coder.decode(pred_bbox_delta[top_n_idx], anchor[top_n_idx]) # 对这些建议框进行解码，得到修正后的建议框
        
        proposal, score = process_box(proposal, score, image_shape, self.min_size)  # 将建议框减小至图像大小，去除过小的建议框
        keep = nms(proposal, score, self.nms_thresh)[:post_nms_top_n]   # 选择前k个建议框
        proposal = proposal[keep]
        return proposal
    
    # 损失函数
    # 输入：
        # objectness: 一个包含所有anchors的二分类分数，表示是否包含有目标
        # pred_bbox_delta: 预测的bounding box偏移量
        # gt_box: 一个包含所有真实标签（ground truth）bounding box的tensor
        # anchor: 一个包含所有anchors的tensor
    # 返回：
        # objectness_loss: 分类损失，用于度量二分类分数的预测值和真实标签之间的差异
        # box_loss: 回归损失，用于度量bounding box偏移量的预测值和真实标签之间的差异
    def compute_loss(self, objectness, pred_bbox_delta, gt_box, anchor):
        iou = box_iou(gt_box, anchor)
        label, matched_idx = self.proposal_matcher(iou)
        
        pos_idx, neg_idx = self.fg_bg_sampler(label)
        idx = torch.cat((pos_idx, neg_idx))
        regression_target = self.box_coder.encode(gt_box[matched_idx[pos_idx]], anchor[pos_idx])
        
        objectness_loss = F.binary_cross_entropy_with_logits(objectness[idx], label[idx])
        box_loss = F.l1_loss(pred_bbox_delta[pos_idx], regression_target, reduction='sum') / idx.numel()

        return objectness_loss, box_loss
        
    # 实现RPN网络的前向传播
    def forward(self, feature, image_shape, target=None):
        if target is not None:
            gt_box = target['boxes']
        anchor = self.anchor_generator(feature, image_shape)
        
        objectness, pred_bbox_delta = self.head(feature)
        objectness = objectness.permute(0, 2, 3, 1).flatten()
        pred_bbox_delta = pred_bbox_delta.permute(0, 2, 3, 1).reshape(-1, 4)

        proposal = self.create_proposal(anchor, objectness.detach(), pred_bbox_delta.detach(), image_shape)
        if self.training:
            objectness_loss, box_loss = self.compute_loss(objectness, pred_bbox_delta, gt_box, anchor)
            return proposal, dict(rpn_objectness_loss=objectness_loss, rpn_box_loss=box_loss)
        
        return proposal, {}