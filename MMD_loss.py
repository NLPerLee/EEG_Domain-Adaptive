import torch
import torch.nn as nn
import torch.optim as optim

# 定义 MMD 损失类
class MMD_loss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss

# 定义简单的特征提取器
class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeatureExtractor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    # 数据准备
    batch_size = 8
    feature_dim = 4

    # 源域数据
    source_data = torch.randn(batch_size, feature_dim).cuda()  # 源域特征向量
    source_labels = torch.randint(0, 2, (batch_size,)).cuda()  # 源域标签

    # 目标域数据
    target_data = torch.randn(batch_size, feature_dim).cuda()  # 目标域特征向量
    target_labels = torch.randint(0, 2, (batch_size,)).cuda()  # 目标域标签

    # 初始化模型和损失函数
    input_dim = 4
    hidden_dim = 64
    output_dim = 4
    num_classes = 2

    feature_extractor = FeatureExtractor(input_dim, hidden_dim, output_dim).cuda()
    classifier = nn.Linear(output_dim, num_classes).cuda()
    mmd_loss_fn = MMD_loss(kernel_type='rbf').cuda()
    classification_loss_fn = nn.CrossEntropyLoss()

    # 优化器
    optimizer = optim.Adam(
        list(feature_extractor.parameters()) + list(classifier.parameters()),
        lr=0.001
    )

    # 训练过程
    num_epochs = 50
    for epoch in range(num_epochs):
        # 提取特征
        source_features = feature_extractor(source_data)
        target_features = feature_extractor(target_data)

        # 分类损失（仅在源域上计算）
        logits_source = classifier(source_features)
        classification_loss = classification_loss_fn(logits_source, source_labels)

        # 领域对齐损失（MMD 损失）
        mmd_loss = mmd_loss_fn(source_features, target_features)

        # 总损失
        total_loss = classification_loss + mmd_loss

        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # 打印损失
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Classification Loss: {classification_loss.item():.4f}, "
                  f"MMD Loss: {mmd_loss.item():.4f}")