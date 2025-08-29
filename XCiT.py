import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import torch.nn.init as init
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset,random_split
from torch.cuda.amp import GradScaler, autocast
import os
import warnings
import time
import h5py
import torch
from multiprocessing import set_start_method
from torch.utils.data import Dataset
import time
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau,ExponentialLR
from sklearn.metrics import roc_curve, roc_auc_score


# 三个感受野特征混合
class FeatureFusion(nn.Module):
    def __init__(self, in_channels=32, num_features=2, dropout_rate=0.5):
        super(FeatureFusion, self).__init__()

        # 全局平均池化层
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 全连接层
        self.dense1 = nn.Linear(in_channels * num_features, in_channels)
        self.dense2 = nn.Linear(in_channels, num_features)

        # Dropout 层
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)


    def forward(self, x1, x2):
        # 全局平均池化
        x1_gap = self.global_avg_pool(x1).view(x1.size(0), -1)  # (batch_size, 16)
        x2_gap = self.global_avg_pool(x2).view(x2.size(0), -1)  # (batch_size, 16)
        # x3_gap = self.global_avg_pool(x3).view(x3.size(0), -1)  # (batch_size, 16)
        # x4_gap = self.global_avg_pool(x4).view(x4.size(0), -1)  # (batch_size, 16)

        # 特征图拼接
        concat_gap = torch.cat([x1_gap, x2_gap], dim=1)  # (batch_size, 64)

        # 全连接层处理
        dense_out = F.sigmoid(self.dense1(concat_gap))  # (batch_size, 16)
        dense_out = self.dropout1(dense_out)  # 添加 Dropout 层
        dense_out = self.dense2(dense_out)  # (batch_size, 4)
        dense_out = self.dropout2(dense_out)  # 添加 Dropout 层

        # 生成权重
        weights = F.softmax(dense_out, dim=1)  # 使用 Softmax 激活函数
        weights = weights.unsqueeze(-1).unsqueeze(-1)  # (batch_size, 4, 1, 1)

        # 加权融合
        fused_feature = x1 * weights[:, 0:1, :, :] + x2 * weights[:, 1:2, :, :] #+ x3 * weights[:, 2:3, :, :] # + x4 * weights[:, 3:4, :, :]

        return fused_feature

# 空间注意力层
class EnhancedSpatialAttention(nn.Module):
    def __init__(self, channels, num_parts=4):
        super(EnhancedSpatialAttention, self).__init__()
        self.channels = channels
        self.num_parts = num_parts
        self.part_channels = channels // num_parts

        # Spatial pooling
        self.pool_h_avg = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w_avg = nn.AdaptiveAvgPool2d((1, None))
        self.pool_h_max = nn.AdaptiveMaxPool2d((None, 1))
        self.pool_w_max = nn.AdaptiveMaxPool2d((1, None))

        # Feature extraction
        self.conv_h = nn.Conv2d(self.part_channels, self.part_channels, kernel_size=(3, 2), stride=1, padding=(1, 0))
        self.conv_w = nn.Conv2d(self.part_channels, self.part_channels, kernel_size=(2, 3), stride=1, padding=(0, 1))

        # Expansion and residual connection
        self.conv_expand = nn.Conv2d(self.part_channels, self.part_channels, kernel_size=1, stride=1, padding=0)

        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, h, w = x.size()

        # Split the feature map into parts along the channel dimension
        parts = torch.split(x, self.part_channels, dim=1)

        # Process each part separately
        attention_maps = []
        for part in parts:
            # Spatial pooling
            part_h_avg = self.pool_h_avg(part)  # b, part_channels, h, 1
            part_w_avg = self.pool_w_avg(part)  # b, part_channels, 1, w
            part_h_max = self.pool_h_max(part)  # b, part_channels, h, 1
            part_w_max = self.pool_w_max(part)  # b, part_channels, 1, w

            # Concatenate average and max pooling results along the last dimension
            part_h = torch.cat([part_h_avg, part_h_max], dim=-1)  # b, part_channels, h, 2
            part_w = torch.cat([part_w_avg, part_w_max], dim=-2)  # b, part_channels, 2, w

            # Feature extraction
            part_h = self.conv_h(part_h).squeeze(-1)  # b, part_channels, h
            part_w = self.conv_w(part_w).squeeze(-2)  # b, part_channels, w

            part_h = self.relu(part_h)
            part_w = self.relu(part_w)

            # Compute attention maps
            part_h = self.softmax(part_h)
            part_w = self.softmax(part_w)

            # Apply attention maps
            part_h = part_h.unsqueeze(-1)  # b, part_channels, h, 1
            part_w = part_w.unsqueeze(-2)  # b, part_channels, 1, w

            attention_map = torch.matmul(part_h, part_w)  # b, part_channels, h, w
            attention_map = self.conv_expand(attention_map)  # b, part_channels, h, w

            attention_maps.append(attention_map)

        # Combine the attention maps back into a single feature map
        out = torch.cat(attention_maps, dim=1)  # b, c, h, w

        # Residual connection
        out = x + x * out.sigmoid()

        return out


# # 通道注意力层
# class ChannelAttentionLayer(nn.Module):
#     def __init__(self, kernel_size=3):
#         super().__init__()
#         self.maxpool = nn.AdaptiveMaxPool2d(1)
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.conv1 = nn.Conv1d(2, 2, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
#         self.conv2 = nn.Conv1d(2, 1, kernel_size=1)
#         self.sigmoid = nn.Sigmoid()
#
#     # 前向传播
#     def forward(self, x):
#         b, c, _, _ = x.size()
#         # 求最大
#         max_result = self.maxpool(x) # b,c,1,1
#         # 求平均
#         avg_result = self.avgpool(x) # b,c,1,1
#         y1 = max_result.squeeze(-1) # b,c,1
#         y2 = avg_result.squeeze(-1) # b,c,1
#         feature = torch.cat((y1, y2), -1)  # (b, c, 2)
#         z = feature.permute(0, 2, 1) # (b,2,c)
#         z = self.conv1(z)  # b,2,c
#         z = self.conv2(z)  # b,1,c
#         z = self.sigmoid(z)  # b,1,c
#         trans = z.permute(0, 2, 1) # b,c,1
#         z = trans.unsqueeze(-1) # b,c,1,1
#
#         return x * z.expand_as(x)

class ChannelAttentionLayer(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.gap(x)  # bs,c,1,1
        y = y.squeeze(-1).permute(0, 2, 1)  # bs,1,c
        y = self.conv(y)  # bs,1,c
        y = self.sigmoid(y)  # bs,1,c
        y = y.permute(0, 2, 1).unsqueeze(-1)  # bs,c,1,1
        return x * y.expand_as(x)




class T_Multi_scale_mixattention(nn.Module):
    def __init__(self, num_channels, dropout_rate=0.25):
        super(T_Multi_scale_mixattention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=(1, 3), stride=1, padding=(0, 1), dilation=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=(1, 3), stride=1, padding=(0, 2), dilation=(1, 2))
        self.conv3 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=(1, 3), stride=1, padding=(0, 5), dilation=(1, 5))
        self.attention = ChannelAttentionLayer()
        self.fusion = FeatureFusion(in_channels=num_channels, num_features=2, dropout_rate=dropout_rate)
        self.bn = nn.BatchNorm2d(num_channels)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout2d(p=dropout_rate)

    # 前向传播
    def forward(self, x):
        x1 = self.bn(self.conv1(x))
        x1 = self.dropout(x1)  # 添加 Dropout 层
        # x2 = self.bn(self.conv2(x))
        # x2 = self.dropout(x2)  # 添加 Dropout 层
        x3 = self.bn(self.conv3(x))
        x3 = self.dropout(x3)  # 添加 Dropout 层
        # y1 = self.attention(x1)
        # y2 = self.attention(x2)
        # y3 = self.attention(x3)
        #y4 = self.attention(x)


        # # 计算混合注意力权重
        # z = z1 + z2 + z3  # b,c,1
        #
        # # 应用 Sigmoid 函数
        # g = self.sigmoid(z)  # b,c,1
        # g = g.unsqueeze(-1)  # b,c,1,1
        # y4 = self.bn(x * g.expand_as(x))
        # y4 = self.dropout(y4)  # 添加 Dropout 层
        feature = self.fusion(x1, x3)


        return feature

class S_Multi_scale_mixattention(nn.Module):
    def __init__(self, num_channels, dropout_rate=0.25):
        super(S_Multi_scale_mixattention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=(3, 1), stride=1, padding=(1, 0), dilation=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=(3, 1), stride=1, padding=(2, 0), dilation=(2, 1))
        self.conv3 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=(3, 1), stride=1, padding=(5, 0), dilation=(5, 1))
        self.attention = ChannelAttentionLayer()
        self.fusion = FeatureFusion(in_channels=num_channels, num_features=2, dropout_rate=dropout_rate)
        self.bn = nn.BatchNorm2d(num_channels)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout2d(p=dropout_rate)

    # 前向传播
    def forward(self, x):
        x1 = self.bn(self.conv1(x))
        x1 = self.dropout(x1)  # 添加 Dropout 层
        # x2 = self.bn(self.conv2(x))
        # x2 = self.dropout(x2)  # 添加 Dropout 层
        x3 = self.bn(self.conv3(x))
        x3 = self.dropout(x3)  # 添加 Dropout 层
        # y1 = self.attention(x1)
        # y2 = self.attention(x2)
        # y3 = self.attention(x3)
        y4 = self.attention(x)

        # # 计算混合注意力权重
        # z = z1 + z2 + z3  # b,c,1
        #
        # # 应用 Sigmoid 函数
        # g = self.sigmoid(z)  # b,c,1
        # g = g.unsqueeze(-1)  # b,c,1,1
        # y4 = self.bn(x * g.expand_as(x))
        # y4 = self.dropout(y4)  # 添加 Dropout 层
        feature = self.fusion(x1, x3)

        return feature

# 定义3个尺度，卷积核大小分别是3、5、7
class Temporal_Scale_block(nn.Module):
    def __init__(self,scale):
        super(Temporal_Scale_block, self).__init__()

        # 卷积层和批量归一化层
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1,scale), stride=1, padding=(0,int((scale-1)/2)))
        self.bn1 = nn.BatchNorm2d(16)
        self.dropout1 = nn.Dropout(p=0.25)  # p 是丢弃概率，这里是 25%
        self.relu1 = nn.ReLU(inplace=False)  # 使用 inplace=False
        self.pooling = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, scale), stride=1,
                               padding=(0, int((scale - 1) / 2)))
        self.bn2 = nn.BatchNorm2d(16)
        self.dropout2 = nn.Dropout(p=0.25)  # p 是丢弃概率，这里是 25%
        self.relu2 = nn.ReLU(inplace=False)  # 使用 inplace=False
        self.pooling = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.multi_feature = T_Multi_scale_mixattention(num_channels = 16)
        self.esa = EnhancedSpatialAttention(channels=16)



    def forward(self,x):
        x = self.relu1(self.bn1(self.conv1(x)))  # 应用 BN 层
        x = self.dropout1(x)  # 应用 Dropout
        x = self.pooling(x)

        x = self.relu2(self.bn2(self.conv2(x)))  # 应用 BN 层
        x = self.dropout2(x)  # 应用 Dropout
        x = self.pooling(x)
        x = self.multi_feature(x)
        x = self.esa(x)

        return x

# 定义3个尺度，卷积核大小分别是3、5、7
class Spatial_Scale_block(nn.Module):
    def __init__(self,scale):
        super(Spatial_Scale_block, self).__init__()

        # 卷积层和批量归一化层
        self.conv1 = nn.Conv2d(in_channels=48, out_channels=32, kernel_size=(scale,1), stride=1, padding=(int((scale-1)/2),0))
        self.bn1 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout(p=0.25)  # p 是丢弃概率，这里是 25%
        self.relu1 = nn.ReLU(inplace=False)  # 使用 inplace=False
        self.pooling = nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(scale,1), stride=1,
                               padding=(int((scale - 1) / 2),0))
        self.bn2 = nn.BatchNorm2d(32)
        self.dropout2 = nn.Dropout(p=0.25)  # p 是丢弃概率，这里是 25%
        self.relu2 = nn.ReLU(inplace=False)  # 使用 inplace=False
        self.pooling = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.multi_feature = S_Multi_scale_mixattention(num_channels = 32)
        self.esa = EnhancedSpatialAttention(channels=32)



    def forward(self,x):
        x = self.relu1(self.bn1(self.conv1(x)))  # 应用 BN 层
        x = self.dropout1(x)  # 应用 Dropout
        x = self.pooling(x)

        x = self.relu2(self.bn2(self.conv2(x)))  # 应用 BN 层
        x = self.dropout2(x)  # 应用 Dropout
        x = self.pooling(x)
        x = self.multi_feature(x)
        x = self.esa(x)

        return x


# 原始XCA实现（来自论文）
class XCA(nn.Module):
    """ Cross-Covariance Attention (XCA) operation where the channels are updated using a weighted
     sum. The weights are obtained from the (softmax normalized) Cross-covariance
    matrix (Q^T K \\in d_h \\times d_h)
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))  # 温度参数（可学习）

        # QKV投影（共享权重）
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        # 输出投影
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape  # B:批次, N:序列长度, C:嵌入维度
        # 生成QKV并拆分
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)  # (B, N, 3, H, C/H)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, C/H)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 分别取Q、K、V，形状均为 (B, H, N, C/H)

        # 转置维度：将通道维度放在序列维度前（核心：计算通道间的交叉协方差）
        q = q.transpose(-2, -1)  # (B, H, C/H, N)
        k = k.transpose(-2, -1)  # (B, H, C/H, N)
        v = v.transpose(-2, -1)  # (B, H, C/H, N)

        # 特征归一化（替代去均值，稳定注意力计算）
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # 计算交叉协方差注意力（通道维度）
        attn = (q @ k.transpose(-2, -1)) * self.temperature  # (B, H, C/H, C/H)，乘温度参数缩放
        attn = attn.softmax(dim=-1)  # 沿通道维度归一化
        attn = self.attn_drop(attn)  # 注意力 dropout

        # 应用注意力到V，并恢复形状
        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)  # (B, N, C)
        x = self.proj(x)  # 输出投影
        x = self.proj_drop(x)  # 输出 dropout
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature'}  # 温度参数不参与权重衰减


# 用原始XCA替换多头注意力的XCiT块
class XCiTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.1):
        super().__init__()
        self.xca = XCA(
            dim=embed_dim,
            num_heads=num_heads,
            attn_drop=dropout,
            proj_drop=dropout
        )
        self.norm1 = nn.LayerNorm(embed_dim)  # 注意力后归一化
        self.norm2 = nn.LayerNorm(embed_dim)  # MLP后归一化

        # MLP部分保持不变
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # XCA注意力 + 残差连接
        x = x + self.xca(self.norm1(x))  # 先归一化再做注意力（与原Transformer一致）
        # MLP + 残差连接
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbedding(nn.Module):
    """Converts feature maps into patches and adds position encoding."""

    def __init__(self, in_channels=96, patch_size=(1, 32), embed_dim=768, img_size=(5, 320)):
        super(PatchEmbedding, self).__init__()
        # Calculate the number of patches
        h_patches = img_size[0] // patch_size[0]
        w_patches = img_size[1] // patch_size[1]
        self.num_patches = h_patches * w_patches

        # Create a learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Convolution to project patches into embedding space
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Positional encoding for each patch
        self.positional_encoding = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

    def forward(self, x):
        B, C, H, W = x.shape
        # Convert to patches and flatten
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # [B, D, H', W'] -> [B, N, D]

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional encoding
        x = x + self.positional_encoding[:, :x.size(1)]
        return x


class CombinedModel(nn.Module):
    def __init__(self, num_classes=2, patch_size=(1, 32), embed_dim=768, num_heads=12, hidden_dim=3072,
                 transformer_layers=3):
        super(CombinedModel, self).__init__()

        # Assuming Temporal_Scale_block and Spatial_Scale_block are already defined elsewhere
        self.temporal_Scale_1 = Temporal_Scale_block(scale=3)
        self.temporal_Scale_2 = Temporal_Scale_block(scale=5)
        self.temporal_Scale_3 = Temporal_Scale_block(scale=7)
        self.bn1 = nn.BatchNorm2d(48)

        self.spatial_Scale_1 = Spatial_Scale_block(scale=3)
        self.spatial_Scale_2 = Spatial_Scale_block(scale=5)
        self.spatial_Scale_3 = Spatial_Scale_block(scale=7)
        self.bn2 = nn.BatchNorm2d(96)

        # Patch Embedding
        self.patch_embedding = PatchEmbedding(in_channels=96, patch_size=patch_size, embed_dim=embed_dim,
                                              img_size=(5, 320))

        # 用原始XCA组成的XCiT块替代Transformer块
        self.transformer_blocks = nn.ModuleList([
            XCiTBlock(embed_dim, num_heads, hidden_dim) for _ in range(transformer_layers)
        ])

        # Classification Head
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # Temporal and Spatial Feature Extraction
        x1 = self.temporal_Scale_1(x)
        x2 = self.temporal_Scale_2(x)
        x3 = self.temporal_Scale_3(x)

        x = torch.cat((x1, x2, x3), dim=1)
        x = self.bn1(x)

        x1 = self.spatial_Scale_1(x)
        x2 = self.spatial_Scale_2(x)
        x3 = self.spatial_Scale_3(x)
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.bn2(x)

        # Convert features to patches and add positional encoding
        x = self.patch_embedding(x)

        # Pass through Transformer layers
        for block in self.transformer_blocks:
            x = block(x)

        # Take the cls_token output for classification
        x = x[:, 0]  # [B, N, D] -> [B, D]

        # Classification head
        x = self.fc(x)  # [B, D] -> [B, num_classes]

        return x



# 定义脑电数据集
class EEGDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['inter', 'pre']
        self.files = []
        self.labels = []

        for cls in self.classes:
            class_dir = os.path.join(root_dir, cls)
            if os.path.isdir(class_dir):
                for eeg_file in os.listdir(class_dir):
                    if eeg_file.endswith('.npy'):
                        self.files.append(os.path.join(class_dir, eeg_file))
                        self.labels.append(self.classes.index(cls))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        eeg_path = self.files[idx]
        data = np.load(eeg_path).astype(np.float32)

        # 升维操作
        data = np.expand_dims(data, axis=0)  # 形状变为 (1, 21, 1280)

        # 标准归一化
        mean = np.mean(data, axis=(1, 2), keepdims=True)
        std = np.std(data, axis=(1, 2), keepdims=True)
        data = (data - mean) / (std + 1e-8)  # 防止除以零

        if self.transform:
            data = self.transform(data)

        label = self.labels[idx]

        return torch.from_numpy(data), label



def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# 实例化模型
model = CombinedModel()

# 将模型移动到 GPU（如果可用）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
# # 定义损失函数
# criterion = nn.BCELoss()  # 二元交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

# # 创建 StepLR 调度器
scheduler = StepLR(optimizer, step_size=100, gamma=0.9)  # 每 10 个 epoch 学习率乘以 0.9
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=4, min_lr=1e-6)
# scheduler = ExponentialLR(optimizer,gamma=0.95)

# 用于保存训练过程中的损失值和准确率
train_losses = []
train_accuracies = []
best_model_weights = None
best_accuracy = 0.0
val_losses = []
val_accuracies = []


def train(num_epochs=50):
    global train_losses, train_accuracies, val_losses, val_accuracies, best_model_weights, best_accuracy
    writer = SummaryWriter('runs/experiment_1')  # 创建一个 TensorBoard writer

    for epoch in range(num_epochs):
        print(f"Epoch [{epoch + 1}/{num_epochs}] - 开始训练")
        start_time = time.time()
        model.train()
        correct = 0
        total_samples = 0
        train_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            # # print(labels.shape)
            # # 确保标签的形状与输出的形状一致
            # labels = labels.unsqueeze(1) # 将标签的形状从 [32] 变为 [32, 1]
            # labels = labels.float()
            # # print(labels.shape)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # 计算训练准确率
            # predicted = (outputs > 0.5).float()  # 使用 0.5 作为阈值
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct += (predicted == labels).sum().item()


        # 打印每个 epoch 的平均损失和训练准确率
        train_loss /= len(train_loader)  # 基于 batch 数量归一化
        train_acc = 100 * correct / total_samples
        end_time = time.time()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%')
        print(end_time - start_time)

        # 保存训练损失和准确率
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # 在验证集上评估模型
        val_loss, val_acc = validate()
        print(f"Val_acc:{val_acc}")
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_model_weights = model.state_dict()
            torch.save(best_model_weights, 'E:\IEEE SENSORS JOURNAL\data\\esa\chb6\\best_model_weights.pth')
            print(f"New best validation accuracy: {val_acc:.2f}% - Model weights saved.")


        # 保存验证集的损失和准确率
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # 将训练和验证的损失和准确率写入 TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        # 获取当前学习率并打印
        current_lr = get_lr(optimizer)
        print(f'Current Learning Rate: {current_lr}')

        # 更新学习率
        scheduler.step()
        # scheduler.step()

    print('Finished Training')
    writer.close()  # 关闭 TensorBoard writer


def validate():
    model.eval()
    correct = 0
    total_samples = 0
    val_loss = 0.0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # labels = labels.unsqueeze(1)  # 将标签的形状从 [32] 变为 [32, 1]
            # labels = labels.float()

            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            # predicted = (outputs > 0.5).float()  # 使用 0.5 作为阈值
            total_samples += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total_samples
    val_loss /= len(val_loader)  # 基于 batch 数量归一化
    return val_loss, val_acc

def test():
    global test_metrics
    model.load_state_dict(best_model_weights)  # 加载最佳权重
    model.eval()
    all_labels = []
    all_predicted = []
    all_outputs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predicted.extend(predicted.cpu().numpy())
            all_outputs.extend(outputs[:, 1].cpu().numpy())  # 假设是二分类问题，取正类的概率

    all_labels = np.array(all_labels)
    all_predicted = np.array(all_predicted)
    all_outputs = np.array(all_outputs)

    # 计算各种指标
    accuracy = 100 * (all_labels == all_predicted).mean()
    precision = precision_score(all_labels, all_predicted)
    recall = recall_score(all_labels, all_predicted)
    auc = roc_auc_score(all_labels, all_outputs)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_predicted).ravel()
    specificity = tn / (tn + fp)

    # 保存测试集的指标
    test_metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'auc': auc,
        'specificity': specificity
    }

    # 打印测试集的指标
    print(f'Accuracy of the network on the test images: {accuracy:.2f}%')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'AUC: {auc:.4f}')
    print(f'Specificity: {specificity:.4f}')

    # 将测试集的指标写入 TensorBoard
    writer = SummaryWriter('runs/experiment_1')
    writer.add_scalar('Accuracy/test', accuracy, 0)
    writer.add_scalar('Precision/test', precision, 0)
    writer.add_scalar('Recall/test', recall, 0)
    writer.add_scalar('AUC/test', auc, 0)
    writer.add_scalar('Specificity/test', specificity, 0)
    writer.close()

    # 计算 ROC 曲线
    fpr, tpr, thresholds = roc_curve(all_labels, all_outputs)

    # 绘制 ROC 曲线
    plt.figure(figsize=(10, 5))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('E:\IEEE SENSORS JOURNAL\data\\esa\chb6\\roc_curve_10.png')
    plt.show()

    # 保存 ROC 曲线数据
    roc_data = np.column_stack((fpr, tpr, thresholds))
    np.savetxt('E:\IEEE SENSORS JOURNAL\data\\esa\chb6\\roc_data_10.txt', roc_data, delimiter=',', header='FPR,TPR,Thresholds')

def plot_and_save_results():
    # 绘制训练和验证损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.savefig('E:\IEEE SENSORS JOURNAL\data\\esa\chb6\\loss_curve_10.png')
    plt.show()

    # 绘制训练和验证准确率曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy over Epochs')
    plt.legend()
    plt.savefig('E:\data\\esa\chb6\\accuracy_curve_10.png')
    plt.show()

if __name__ == "__main__":
    # 设置多处理模式（可选，但推荐）
    set_start_method('spawn')

    # 数据路径
    root_train_folder = "E:\IEEE SENSORS JOURNAL\\raw_2\\train_spilt\chb6"
    root_test_folder = "E:\IEEE SENSORS JOURNAL\\raw_2\\test_spilt\chb6"
    # 创建数据集
    dataset = EEGDataset(root_train_folder)
    test_dataset = EEGDataset(root_test_folder)

    # 划分数据集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 创建数据加载器
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=8,pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=8,pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers=8,pin_memory=True)

    train()
    test()

    # 保存训练过程中的损失值和准确率
    with open('E:\IEEE SENSORS JOURNAL\data\\esa\chb6\\train_chb3.txt', 'w') as f:
        f.write('Epoch,Train Loss,Train Accuracy,Val Loss,Val Accuracy\n')
        for epoch, (t_loss, t_acc, v_loss, v_acc) in enumerate(
                zip(train_losses, train_accuracies, val_losses, val_accuracies)):
            f.write(f'{epoch + 1},{t_loss:.4f},{t_acc:.2f},{v_loss:.4f},{v_acc:.2f}\n')

    # 保存测试集的指标
    with open('E:\IEEE SENSORS JOURNAL\data\\esa\chb6\\test_chb3.txt', 'w') as f:
        f.write('Metric,Value\n')
        for metric, value in test_metrics.items():
            f.write(f'{metric},{value:.4f}\n')

    # 绘制并保存结果
    plot_and_save_results()
