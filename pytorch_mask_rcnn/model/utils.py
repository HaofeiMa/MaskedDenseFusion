import torch


class Matcher:
    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, iou):
        """
        Arguments:
            iou (Tensor[M, N]): containing the pairwise quality between 
            M ground-truth boxes and N predicted boxes.

        Returns:
            label (Tensor[N]): positive (1) or negative (0) label for each predicted box,
            -1 means ignoring this box.
            matched_idx (Tensor[N]): indices of gt box matched by each predicted box.
        """
        
        value, matched_idx = iou.max(dim=0)
        label = torch.full((iou.shape[1],), -1, dtype=torch.float, device=iou.device) 
        
        label[value >= self.high_threshold] = 1
        label[value < self.low_threshold] = 0
        
        if self.allow_low_quality_matches:
            highest_quality = iou.max(dim=1)[0]
            gt_pred_pairs = torch.where(iou == highest_quality[:, None])[1]
            label[gt_pred_pairs] = 1

        return label, matched_idx
    

class BalancedPositiveNegativeSampler:
    def __init__(self, num_samples, positive_fraction):
        self.num_samples = num_samples
        self.positive_fraction = positive_fraction

    def __call__(self, label):
        positive = torch.where(label == 1)[0]
        negative = torch.where(label == 0)[0]

        num_pos = int(self.num_samples * self.positive_fraction)
        num_pos = min(positive.numel(), num_pos)
        num_neg = self.num_samples - num_pos
        num_neg = min(negative.numel(), num_neg)

        pos_perm = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
        neg_perm = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

        pos_idx = positive[pos_perm]
        neg_idx = negative[neg_perm]

        return pos_idx, neg_idx

    
def roi_align(features, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio):
    if torch.__version__ >= "1.5.0":
        return torch.ops.torchvision.roi_align(
            features, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio, False)
    else:
        return torch.ops.torchvision.roi_align(
            features, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio)


class AnchorGenerator:
    def __init__(self, sizes, ratios):
        self.sizes = sizes
        self.ratios = ratios
        
        self.cell_anchor = None
        self._cache = {}
        
    # 为每个特征图单元设置一个默认的Anchor框，这些框基于size和ratios组合而来
    def set_cell_anchor(self, dtype, device):
        if self.cell_anchor is not None:
            return 
        sizes = torch.tensor(self.sizes, dtype=dtype, device=device)
        ratios = torch.tensor(self.ratios, dtype=dtype, device=device)

        h_ratios = torch.sqrt(ratios)   # 计算宽高比的平方根，得到高度比率
        w_ratios = 1 / h_ratios         # 得到宽度比率

        hs = (sizes[:, None] * h_ratios[None, :]).view(-1)
        ws = (sizes[:, None] * w_ratios[None, :]).view(-1)

        self.cell_anchor = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        
    # 生成一组在特征图上的锚点，用于生成候选框
    #   grid_size：一个长度为2的元组或列表，表示特征图的大小，如(38, 50)
    #   stride：一个长度为2的元组或列表，表示在输入图片上滑动特征图时的步长，如(16, 16)
    def grid_anchor(self, grid_size, stride):
        dtype, device = self.cell_anchor.dtype, self.cell_anchor.device
        # 在水平方向上，生成大小为 grid_size[1] 的一维张量 shift_x，每个元素的值等于它的下标乘以步长stride[1]，表示在x轴上每个cell左上角的x坐标。
        shift_x = torch.arange(0, grid_size[1], dtype=dtype, device=device) * stride[1]
        # 在竖直方向上，同理生成大小为 grid_size[0] 的一维张量 shift_y，表示在y轴上每个cell左上角的y坐标。
        shift_y = torch.arange(0, grid_size[0], dtype=dtype, device=device) * stride[0]

        # 然后通过这两个张量生成两个grid网格，每个网格上的值分别表示每个cell左上角的x坐标和y坐标。
        y, x = torch.meshgrid(shift_y, shift_x)
        x = x.reshape(-1)
        y = y.reshape(-1)
        # 将这两个网格拼接起来，得到一个大小为（grid_size[0] * grid_size[1]）×2的二维张量，每行代表一个cell的左上角坐标。
        # 将这个二维张量的每个元素都复制成一个大小为(1, 1, 4)的三维张量
        shift = torch.stack((x, y, x, y), dim=1).reshape(-1, 1, 4)

        # 然后将每个三维张量加上锚点，得到一组大小为(grid_size[0] * grid_size[1] * k, 4)的锚点，其中k为每个cell中锚点的数量。
        anchor = (shift + self.cell_anchor).reshape(-1, 4)
        # 返回这组在特征图上的锚点
        return anchor
        
    # 实现Anchor的缓存，通过一个字典_cache来保存已经计算过的Anchor框，当grid_size和stride已经存在是，直接返回结果，否则计算新的结果
    def cached_grid_anchor(self, grid_size, stride):
        key = grid_size + stride
        if key in self._cache:
            return self._cache[key]
        anchor = self.grid_anchor(grid_size, stride)
        
        if len(self._cache) >= 3:
            self._cache.clear()
        self._cache[key] = anchor
        return anchor

    def __call__(self, feature, image_size):
        dtype, device = feature.dtype, feature.device
        grid_size = tuple(feature.shape[-2:])
        stride = tuple(int(i / g) for i, g in zip(image_size, grid_size))
        
        self.set_cell_anchor(dtype, device)
        
        anchor = self.cached_grid_anchor(grid_size, stride)
        return anchor