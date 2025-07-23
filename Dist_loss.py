import torch
import torch.nn as nn
import torch.nn.functional as F


class Dist_Loss(nn.Module):
    def __init__(self, temperature=10.0):
        super(Dist_Loss, self).__init__()
        self.temperature = temperature  # 温度参数，控制距离缩放

    def forward(self, feat1, label1, feat2, label2):
        # 类内紧凑性
        intra_dist = 0.0
        classes = torch.unique(label1)
        for c in classes:
            idx1 = (label1 == c).nonzero(as_tuple=True)[0]
            idx2 = (label2 == c).nonzero(as_tuple=True)[0]
            if len(idx1) > 1 and len(idx2) > 1:
                c_feat1 = feat1[idx1]
                c_feat2 = feat2[idx2]
                center1 = torch.mean(c_feat1, dim=0, keepdim=True)
                center2 = torch.mean(c_feat2, dim=0, keepdim=True)
                intra_dist += torch.mean(torch.cdist(c_feat1, center1))
                intra_dist += torch.mean(torch.cdist(c_feat2, center2))

        # 类间分离性
        inter_dist = 0.0
        centers1, centers2 = [], []
        for c in classes:
            idx1 = (label1 == c).nonzero(as_tuple=True)[0]
            idx2 = (label2 == c).nonzero(as_tuple=True)[0]
            if len(idx1) > 0 and len(idx2) > 0:
                centers1.append(torch.mean(feat1[idx1], dim=0))
                centers2.append(torch.mean(feat2[idx2], dim=0))

        if len(centers1) > 1 and len(centers2) > 1:
            centers1 = torch.stack(centers1)
            centers2 = torch.stack(centers2)
            inter_dist = torch.mean(torch.cdist(centers1, centers2))

        # 温度缩放和归一化
        if inter_dist > 0:
            normalized_dist = intra_dist / (inter_dist + 1e-8)
            # 使用温度缩放避免梯度爆炸
            loss = torch.log(1 + torch.exp(normalized_dist / self.temperature))
        else:
            loss = intra_dist

        return loss


# 示例：定义一个简单的特征提取模型
class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeatureExtractor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 示例验证（保持不变）
if __name__ == "__main__":
    input_dim, hidden_dim, output_dim = 10, 64, 4
    batch_size = 8
    inputs = torch.randn(batch_size, input_dim).to("cuda")
    labels = torch.tensor([0, 1, 0, 1, 2, 2, 3, 3]).to("cuda")

    feature_extractor = FeatureExtractor(input_dim, hidden_dim, output_dim).to("cuda")
    dist_loss_fn = Dist_Loss().to("cuda")
    optimizer = torch.optim.Adam(feature_extractor.parameters(), lr=0.01)

    features = feature_extractor(inputs)
    loss = dist_loss_fn(features, labels, features, labels)  # 假设源域和目标域相同，仅作测试

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("Loss:", loss.item())

