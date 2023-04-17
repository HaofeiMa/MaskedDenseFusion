from collections import OrderedDict

import torch.nn.functional as F
from torch import nn
from torch.utils.model_zoo import load_url
from torchvision import models
from torchvision.ops import misc

from .utils import AnchorGenerator
from .rpn import RPNHead, RegionProposalNetwork
from .pooler import RoIAlign
from .roi_heads import RoIHeads
from .transform import Transformer


class MaskRCNN(nn.Module):
    """
    Implements Mask R-CNN.

    The input image to the model is expected to be a tensor, shape [C, H, W], and should be in 0-1 range.

    The behavior of the model changes depending if it is in training or evaluation mode.
    
    During training, the model expects both the input tensor, as well as a target (dictionary),
    containing:
        - boxes (FloatTensor[N, 4]): the ground-truth boxes in [xmin, ymin, xmax, ymax] format, with values
          between 0-H and 0-W
        - labels (Int64Tensor[N]): the class label for each ground-truth box
        - masks (UInt8Tensor[N, H, W]): the segmentation binary masks for each instance

    The model returns a Dict[Tensor], containing the classification and regression losses 
    for both the RPN and the R-CNN, and the mask loss.

    During inference, the model requires only the input tensor, and returns the post-processed
    predictions as a Dict[Tensor]. The fields of the Dict are as
    follows:
        - boxes (FloatTensor[N, 4]): the predicted boxes in [xmin, ymin, xmax, ymax] format, 
          with values between 0-H and 0-W
        - labels (Int64Tensor[N]): the predicted labels
        - scores (FloatTensor[N]): the scores for each prediction
        - masks (FloatTensor[N, H, W]): the predicted masks for each instance, in 0-1 range. In order to
          obtain the final segmentation masks, the soft masks can be thresholded, generally
          with a value of 0.5 (mask >= 0.5)
        
    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
        num_classes (int): number of output classes of the model (including the background).
        
        rpn_fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        rpn_bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        rpn_num_samples (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        rpn_positive_fraction (float): proportion of positive anchors during training of the RPN
        rpn_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
            bounding boxes
        rpn_pre_nms_top_n_train (int): number of proposals to keep before applying NMS during training
        rpn_pre_nms_top_n_test (int): number of proposals to keep before applying NMS during testing
        rpn_post_nms_top_n_train (int): number of proposals to keep after applying NMS during training
        rpn_post_nms_top_n_test (int): number of proposals to keep after applying NMS during testing
        rpn_nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
        
        box_fg_iou_thresh (float): minimum IoU between the proposals and the GT box so that they can be
            considered as positive during training of the classification head
        box_bg_iou_thresh (float): maximum IoU between the proposals and the GT box so that they can be
            considered as negative during training of the classification head
        box_num_samples (int): number of proposals that are sampled during training of the
            classification head
        box_positive_fraction (float): proportion of positive proposals during training of the 
            classification head
        box_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
            bounding boxes
        box_score_thresh (float): during inference, only return proposals with a classification score
            greater than box_score_thresh
        box_nms_thresh (float): NMS threshold for the prediction head. Used during inference
        box_num_detections (int): maximum number of detections, for all classes.
        
    """

    """
    MaskR-CNN。

    模型的输入图像应该是张量，形状[C，H，W]，并且应该在0-1范围内。

    模型的行为取决于它是处于训练模式还是评估模式。

    在训练期间，模型期望输入张量以及目标（字典），
    包含：
        - boxes (FloatTensor[N, 4]): [xmin，ymin，xmax，ymax]格式的真实边界框，取值在0-H 和 0-W之间
        - labels (Int64Tensor[N]): 每个真实边界框的分类标签
        - masks (UInt8Tensor[N, H, W]): 每个实例中的分段二进制掩码mask
    
    该模型返回Dict[Tensor]，包含RPN和R-CNN的分类和回归损失以及掩码损失。

    在预测过程中，模型只需要输入张量，并将后处理的预测作为Dict[tensor]返回。Dict的字段如下：
        - boxes (FloatTensor[N, 4]): [xmin，ymin，xmax，ymax]格式的真实边界框，取值在0-H 和 0-W之间
        - labels (Int64Tensor[N]): 每个真实边界框的分类标签
        - scores (FloatTensor[N]): 每个预测的分数
        - masks (UInt8Tensor[N, H, W]): 每个实例的预测掩码在0-1范围内。为了获得最终的分割掩模，可以对软掩模进行阈值化，通常值为0.5（掩模>=0.5）

    Arguments:
        backbone (nn.Module): 用于计算模型特征的网络
        num_classes (int): 模型的输出类的数量（包括背景）
        
        rpn_fg_iou_thresh (float): anchor和真实边界框之间的最小IoU，以便在RPN的训练过程中被认为是正值
        rpn_bg_iou_thresh (float): anchor和真实边界框之间的最大IoU，以便在RPN训练期间可以将其视为负值
        rpn_num_samples (int): 在RPN训练期间为计算损失而采样的anchor的数量
        rpn_positive_fraction (float): RPN训练期间，正anchor的比例
        rpn_reg_weights (Tuple[float, float, float, float]): 边界框编码解码的权重
        rpn_pre_nms_top_n_train (int): 训练时使用NMS之前保留的建议数量
        rpn_pre_nms_top_n_test (int): 测试时使用NMS之前保留的建议数量
        rpn_post_nms_top_n_train (int): 训练时使用NMS之后保留的建议数量
        rpn_post_nms_top_n_test (int): 测试时使用NMS之后保留的建议数量
        rpn_nms_thresh (float): 用于RPN建议网络的 NMS 阈值
        
        box_fg_iou_thresh (float): 建议框和真实边界框的最小IOU，以便在分类训练期间将其视为正值
        box_bg_iou_thresh (float): 建议框和真实边界框的最大IOU，以便在分类训练期间将其视为负值
        box_num_samples (int): 分类训练时的建议框数量
        box_positive_fraction (float): 分类训练时的正值建议框比例
        box_reg_weights (Tuple[float, float, float, float]): 边界框编码/解码的权重
        box_score_thresh (float): 预测时，只返回分类分数大于box_score_thresh的建议
        box_nms_thresh (float): 预测时的NMS阈值
        box_num_detections (int): 所有类别的最大检测次数
    """
    
    def __init__(self, backbone, num_classes,
                 # RPN parameters
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_num_samples=256, rpn_positive_fraction=0.5,
                 rpn_reg_weights=(1., 1., 1., 1.),
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 # RoIHeads parameters
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_num_samples=512, box_positive_fraction=0.25,
                 box_reg_weights=(10., 10., 5., 5.),
                 box_score_thresh=0.1, box_nms_thresh=0.6, box_num_detections=100):
        super().__init__()
        self.backbone = backbone
        out_channels = backbone.out_channels
        
        #------------- RPN --------------------------
        anchor_sizes = (128, 256, 512)  # 定义三种大小的anchor
        anchor_ratios = (0.5, 1, 2)     # 定义三种比例的anchor
        num_anchors = len(anchor_sizes) * len(anchor_ratios)    # 一共有九种anchor
        rpn_anchor_generator = AnchorGenerator(anchor_sizes, anchor_ratios) # anchor生成器
        rpn_head = RPNHead(out_channels, num_anchors)
        
        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)
        self.rpn = RegionProposalNetwork(   # 定义RPN区域建议网络
             rpn_anchor_generator, rpn_head, 
             rpn_fg_iou_thresh, rpn_bg_iou_thresh,
             rpn_num_samples, rpn_positive_fraction,
             rpn_reg_weights,
             rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)
        
        #------------ RoIHeads --------------------------
        # 创建 RoIAlign 对象，用于将 RoI 特征图划分为固定大小的特征图以用于后续处理。
        box_roi_pool = RoIAlign(output_size=(7, 7), sampling_ratio=2)
        
        # 定义 Fast R-CNN 预测器，用于对 RoI 特征图进行分类和回归。
        resolution = box_roi_pool.output_size[0]
        in_channels = out_channels * resolution ** 2
        mid_channels = 1024
        box_predictor = FastRCNNPredictor(in_channels, mid_channels, num_classes)
        
        # 定义 RoIHeads 对象，用于对每个 RoI 特征图执行分类、回归和非极大值抑制 (NMS) 操作，以生成最终的目标框。
        self.head = RoIHeads(
             box_roi_pool, box_predictor,
             box_fg_iou_thresh, box_bg_iou_thresh,
             box_num_samples, box_positive_fraction,
             box_reg_weights,
             box_score_thresh, box_nms_thresh, box_num_detections)
        
        # 为 RoIHeads 对象创建 mask_roi_pool 对象，用于对每个 RoI 特征图执行区域兴趣池化，以生成分割掩膜的特征图。
        self.head.mask_roi_pool = RoIAlign(output_size=(14, 14), sampling_ratio=2)
        
        # 定义 Mask R-CNN 预测器，用于预测每个 RoI 特征图的掩膜。
        layers = (256, 256, 256, 256)
        dim_reduced = 256
        self.head.mask_predictor = MaskRCNNPredictor(out_channels, layers, dim_reduced, num_classes)
        
        #------------ Transformer --------------------------
        self.transformer = Transformer(
            min_size=800, max_size=1333, 
            image_mean=[0.485, 0.456, 0.406], 
            image_std=[0.229, 0.224, 0.225])
        
    # 输入：包含图像数据的张量image，target包含每个目标在图像中的位置和类别等信息
    # 返回：返回字典，如果有target则返回 rpn_losses 和 roi_losses，如果没有target，则返回包含检测结果的字典
    def forward(self, image, target=None):
        ori_image_shape = image.shape[-2:]  # 获取原始图像的大小
        
        image, target = self.transformer(image, target) # 使用transformer对图像进行增强
        image_shape = image.shape[-2:]
        feature = self.backbone(image)  # 将图像输入backbone得到特征图feature
        
        proposal, rpn_losses = self.rpn(feature, image_shape, target)   # 使用 rpn 对 feature 进行前向传播，得到 proposal 和 rpn_losses
        result, roi_losses = self.head(feature, proposal, image_shape, target)  # 将 proposal 和 feature 作为 RoIHeads 的输入，使用 RoIHeads 进行前向传播，得到检测结果 result 和 roi_losses
        
        if self.training:
            return dict(**rpn_losses, **roi_losses)
        else:
            result = self.transformer.postprocess(result, image_shape, ori_image_shape)
            return result
        
        
class FastRCNNPredictor(nn.Module):
    def __init__(self, in_channels, mid_channels, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, mid_channels)
        self.fc2 = nn.Linear(mid_channels, mid_channels)
        self.cls_score = nn.Linear(mid_channels, num_classes)
        self.bbox_pred = nn.Linear(mid_channels, num_classes * 4)
        
    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        score = self.cls_score(x)
        bbox_delta = self.bbox_pred(x)

        return score, bbox_delta        
    
    
class MaskRCNNPredictor(nn.Sequential):
    def __init__(self, in_channels, layers, dim_reduced, num_classes):
        """
        Arguments:
            in_channels (int)
            layers (Tuple[int])
            dim_reduced (int)
            num_classes (int)
        """
        
        d = OrderedDict()
        next_feature = in_channels
        for layer_idx, layer_features in enumerate(layers, 1):
            d['mask_fcn{}'.format(layer_idx)] = nn.Conv2d(next_feature, layer_features, 3, 1, 1)
            d['relu{}'.format(layer_idx)] = nn.ReLU(inplace=True)
            next_feature = layer_features
        
        d['mask_conv5'] = nn.ConvTranspose2d(next_feature, dim_reduced, 2, 2, 0)
        d['relu5'] = nn.ReLU(inplace=True)
        d['mask_fcn_logits'] = nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)
        super().__init__(d)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                
# 搭建带有特征金字塔的ResNet网络
class ResBackbone(nn.Module):
    def __init__(self, backbone_name, pretrained):
        super().__init__()
        # 实例化ResNet网络
        body = models.resnet.__dict__[backbone_name](
            pretrained=pretrained, norm_layer=misc.FrozenBatchNorm2d)
        
        # 冻结在一个 ResNet 模型的 backbone 中除 layer2，layer3 和 layer4 以外的所有层的参数，使这些参数在训练时不更新
        for name, parameter in body.named_parameters():
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
                
        # 将body的前8层作为子模块加入到ModuelDict中，以便后续网络中使用
        self.body = nn.ModuleDict(d for i, d in enumerate(body.named_children()) if i < 8)
        in_channels = 2048
        self.out_channels = 256
        
        # 特征金字塔
        self.inner_block_module = nn.Conv2d(in_channels, self.out_channels, 1)  # 卷积层，将输入特征图通道从2048转换为256
        self.layer_block_module = nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1)
        
        # 对刚定义的两个卷积层权重和偏置初始化
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)
    
    # 实现ResNet特征提取器的前向传播，将输入的特征图依次通过 ResNet 的前8层卷积层，然后经过两个卷积层进行特征金字塔的处理，最终输出处理后的特征图
    def forward(self, x):
        for module in self.body.values():
            x = module(x)
        x = self.inner_block_module(x)
        x = self.layer_block_module(x)
        return x

# 在ResNet50的基础上构建Mask RCNN模型
def maskrcnn_resnet50(pretrained, num_classes, pretrained_backbone=True):
    """
    Constructs a Mask R-CNN model with a ResNet-50 backbone.
    
    Arguments:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017.
        num_classes (int): number of classes (including the background).
    """
    
    if pretrained:
        backbone_pretrained = False
        
    backbone = ResBackbone('resnet50', pretrained_backbone)
    model = MaskRCNN(backbone, num_classes)
    
    # 加载预训练的Mask RCNN的权重
    if pretrained:
        model_urls = {
            'maskrcnn_resnet50_fpn_coco':
                'https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth',
        }
        model_state_dict = load_url(model_urls['maskrcnn_resnet50_fpn_coco'])
        
        pretrained_msd = list(model_state_dict.values())
        del_list = [i for i in range(265, 271)] + [i for i in range(273, 279)]
        for i, del_idx in enumerate(del_list):
            pretrained_msd.pop(del_idx - i)

        msd = model.state_dict()
        skip_list = [271, 272, 273, 274, 279, 280, 281, 282, 293, 294]
        if num_classes == 91:
            skip_list = [271, 272, 273, 274]
        for i, name in enumerate(msd):
            if i in skip_list:
                continue
            msd[name].copy_(pretrained_msd[i])
            
        model.load_state_dict(msd)
    
    return model