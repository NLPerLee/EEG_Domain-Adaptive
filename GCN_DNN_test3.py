import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import shutil
from torch.utils.data import Dataset
from torch.nn import Parameter, BatchNorm1d
import torch
import random
import numpy as np
import os
import time
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.model_selection import train_test_split
import MMD_loss
import Dist_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
NUM_NODES = 21  # EEG通道数
IN_CHANNELS = 48  # 特征维度
HIDDEN_CHANNELS = 64  # 隐藏层维度
OUT_CHANNELS = 128  # 输出维度
NUM_CLASSES = 2  # 分类类别数
NUM_DOMAINS = 2  # 源域数量
BATCH_SIZE = 128  # 总批次大小（每个域的样本数 = BATCH_SIZE // (NUM_DOMAINS+1)）
NUM_EPOCHS = 30  # 训练轮数
LR = 0.001  # 学习率
GAMMA = 0.95  # 学习率衰减因子
LAMBDA_D = 5  # 域对抗损失权重
lambda_mmd = 1.0  # MMD损失权重
lambda_dist = 0.5  # Dist损失权重
alpha = 0.1  # Dist损失中的类间分离性权重


### 设置随机种子,保证复现
def set_seed(seed=42):
    # Python built-in random module
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU

    # Disable cudnn to ensure deterministic results
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 在您的主程序或训练脚本开始时调用此函数
set_seed(42)  # 使用您选择的种子数


# 定义梯度反转层
class GradientReverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambda_
        return output, None


class GradientReverseLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReverseFunction.apply(x, self.lambda_)


# 渐进型梯度反转参数
class WarmStartGradientReverseLayer(nn.Module):
    def __init__(self, alpha: float = 10.0, lo: float = 0.0, hi: float = 1.0,
                 max_iters: int = 1000, auto_step: bool = True):
        super().__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # 增强反转系数的增长速率
        coeff = 2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters)) - (
                    self.hi - self.lo) + self.lo
        if self.auto_step:
            self.step()
        return GradientReverseFunction.apply(input, coeff)

    def step(self):
        self.iter_num += 1

class AdaptiveGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, fixed_adj_matrix=None, num_nodes=21,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(AdaptiveGCN, self).__init__()
        self.num_nodes = num_nodes
        self.device = device

        # 特征映射层
        self.feature_mapper = torch.nn.Linear(in_channels, hidden_channels)

        # 图卷积层
        self.conv1 = GCNConv(hidden_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

        # 批归一化层
        self.bn1 = BatchNorm1d(hidden_channels)  # 直接使用 BatchNorm1d
        self.bn2 = BatchNorm1d(out_channels)

        # 初始化自适应结构参数
        if fixed_adj_matrix is not None:
            # 确保fixed_adj_matrix是张量并在正确设备上
            if not isinstance(fixed_adj_matrix, torch.Tensor):
                fixed_adj_matrix = torch.tensor(fixed_adj_matrix, dtype=torch.float32)
            fixed_adj_matrix = fixed_adj_matrix.to(self.device)

            # 使用SVD初始化
            m, p, n = torch.svd(fixed_adj_matrix)
            rank = min(num_nodes, 10)  # 设置rank为较小的值
            initemb1 = torch.mm(m[:, :rank], torch.diag(p[:rank] ** 0.5))
            initemb2 = torch.mm(torch.diag(p[:rank] ** 0.5), n[:, :rank].t())
            self.nodevec1 = Parameter(initemb1, requires_grad=True)
            self.nodevec2 = Parameter(initemb2, requires_grad=True)
            self.fixed_adj_matrix = fixed_adj_matrix
        else:
            # 随机初始化
            self.nodevec1 = Parameter(torch.randn(num_nodes, 10), requires_grad=True)
            self.nodevec2 = Parameter(torch.randn(10, num_nodes), requires_grad=True)
            self.fixed_adj_matrix = None

        # 保存训练后的结构
        self.register_buffer('learned_edge_index', None)
        self.register_buffer('learned_edge_weight', None)

        self.domain_attn = nn.Linear(out_channels, 1)
    def forward(self, data):
        x = data.x
        batch = data.batch if hasattr(data, 'batch') else None

        # 特征映射
        x_mapped = F.relu(self.feature_mapper(x))

        # 构建自适应邻接矩阵
        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)

        # 合并固定结构和自适应结构
        if self.fixed_adj_matrix is not None:
            combined_adp = adp + self.fixed_adj_matrix
            # 可以考虑其他归一化方法
            # 例如对称归一化: D^(-1/2) * A * D^(-1/2)
            degree = torch.sum(combined_adp, dim=1)
            degree_inv_sqrt = torch.pow(degree, -0.5)
            degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
            degree_matrix_inv_sqrt = torch.diag(degree_inv_sqrt)
            normalized_adp = degree_matrix_inv_sqrt @ combined_adp @ degree_matrix_inv_sqrt
        else:
            normalized_adp = adp

        # 转换为edge_index和edge_weight格式
        adaptive_edge_index = normalized_adp.nonzero().t().contiguous()
        edge_weight = normalized_adp[adaptive_edge_index[0], adaptive_edge_index[1]]

        # 图卷积
        x = F.relu(self.conv1(x_mapped, adaptive_edge_index, edge_weight))
        x = self.bn1(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = F.relu(self.conv2(x, adaptive_edge_index, edge_weight))
        x = self.bn2(x)

        # 全局池化
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long).to(x.device)
        x = global_mean_pool(x, batch)
        attn_weights = torch.sigmoid(self.domain_attn(x))
        domain_specific_features = x * attn_weights  # 增强域相关特征
        return domain_specific_features

    def reset_learned_structure(self):
        """重置学习到的结构，用于重新训练"""
        self.learned_edge_index = None
        self.learned_edge_weight = None

class LabelClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, num_classes)
        self.bn = nn.BatchNorm1d(hidden_channels)

    def forward(self, x):
        x = F.relu(self.bn(self.fc1(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return x

class DomainClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels=256, num_domains=3):
        super().__init__()
        self.grl = WarmStartGradientReverseLayer(alpha=10.0, max_iters=1000)
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels)
        self.fc3 = nn.Linear(hidden_channels, num_domains)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.dropout = nn.Dropout(0.7)  # 增加dropout防止过拟合

    def forward(self, x):
        x = self.grl(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class EEGDANN(nn.Module):
    def __init__(self, in_channels=5, hidden_channels=64, out_channels=128, num_classes=2, num_nodes=21,
                 device='cuda',fixed_adj_matrix=None):

        super().__init__()
        self.feature_extractor = AdaptiveGCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            fixed_adj_matrix=fixed_adj_matrix,  # 明确指定固定邻接矩阵
            num_nodes=num_nodes,
            device=device
        )

        self.label_classifier = LabelClassifier(out_channels, hidden_channels, num_classes)
        self.domain_classifier = DomainClassifier(out_channels, hidden_channels)

    def forward(self, x):
        features = self.feature_extractor(x)
        class_output = self.label_classifier(features)
        domain_output = self.domain_classifier(features)
        return features, class_output, domain_output

def build_adjacency_matrix(adj_list, num_nodes):
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for node, neighbors in adj_list.items():
        for neighbor in neighbors:
            adj_matrix[node, neighbor] = 1
            adj_matrix[neighbor, node] = 1

    np.fill_diagonal(adj_matrix, 1)

    # 归一化处理 (对称归一化)
    degree = np.sum(adj_matrix, axis=1)
    degree_inv_sqrt = np.power(degree, -0.5)
    degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.0  # 处理度为0的节点
    degree_matrix_inv_sqrt = np.diag(degree_inv_sqrt)
    adj_matrix = degree_matrix_inv_sqrt @ adj_matrix @ degree_matrix_inv_sqrt

    return adj_matrix

def load_and_process_eeg_data(folder_path, label, normalize=True):
    data_list = []
    labels_list = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.npy'):
            file_path = os.path.join(folder_path, filename)
            eeg_segment = np.load(file_path)  # 假设形状为 (21, 5)

            if normalize:
                # 按列进行Z-score归一化
                mean = np.mean(eeg_segment, axis=0, keepdims=True)  # 计算每列均值
                std = np.std(eeg_segment, axis=0, keepdims=True)  # 计算每列标准差
                std = np.maximum(std, 1e-8)  # 防止除以零
                normalized_features = (eeg_segment - mean) / std  # 归一化
            else:
                normalized_features = eeg_segment

            data_list.append(normalized_features)
            labels_list.append(label)

    return np.array(data_list), np.array(labels_list)

# 按照域的维度加载数据
def load_domain_data(domain_path):
    pre_data, _ = load_and_process_eeg_data(os.path.join(domain_path, "pre"), 0)
    inter_data, _ = load_and_process_eeg_data(os.path.join(domain_path, "inter"), 1)

    # 合并数据并打乱
    features = np.concatenate([pre_data, inter_data])
    labels = np.concatenate([np.zeros(len(pre_data)), np.ones(len(inter_data))])

    # 关键改进：随机打乱数据
    np.random.seed(SEED)  # 使用全局随机种子
    indices = np.random.permutation(len(features))
    features = features[indices]
    labels = labels[indices]

    return features, labels

class EEGTestDataset(Dataset):
    def __init__(self, features, labels, adj_matrix):
        self.features = features
        self.labels = labels
        self.adj_matrix = adj_matrix

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        data = Data(x=x, y=y, adj=self.adj_matrix)
        return data

def adj_matrix_to_edge_index(adj_matrix):
    """将邻接矩阵转换为edge_index格式"""
    rows, cols = torch.nonzero(adj_matrix, as_tuple=True)
    edge_index = torch.stack([rows, cols], dim=0)
    edge_weight = adj_matrix[rows, cols]
    return edge_index, edge_weight


class MultiDomainDataset(Dataset):
    def __init__(self, source1_path, source2_path, target_path, adj_matrix):
        # 加载各域数据
        s1_feat, s1_label = load_domain_data(source1_path)
        s2_feat, s2_label = load_domain_data(source2_path)
        t_feat, t_label = load_domain_data(target_path)

        # 划分目标域数据
        t_train_feat, t_test_feat, t_train_label, t_test_label = train_test_split(
            t_feat, t_label, test_size=0.2, random_state=42, stratify=t_label
        )

        # 存储数据
        self.source_data = [
            (s1_feat, s1_label, 0),
            (s2_feat, s2_label, 1),
            (t_train_feat, t_train_label, 2)
        ]
        self.target_test = (t_test_feat, t_test_label)

        # 转换邻接矩阵为edge_index
        self.edge_index, self.edge_weight = adj_matrix_to_edge_index(adj_matrix)

    def __getitem__(self, idx):
        samples = []
        for feat, label, domain in self.source_data:
            # 处理索引越界
            sample_idx = idx % len(feat)
            x = torch.tensor(feat[sample_idx], dtype=torch.float32)
            y = torch.tensor(int(label[sample_idx]), dtype=torch.long)

            # 正确初始化Data对象，使用edge_index和edge_weight
            data = Data(
                x=x,
                y=y,
                edge_index=self.edge_index,
                edge_weight=self.edge_weight,
                domain=domain
            )
            samples.append(data)
        return samples

    def __len__(self):
        return max(len(d[0]) for d in self.source_data)

# 自定义collate函数，将同一域的样本合并为一个Batch
def collate_fn(batch):
    """多域数据批处理函数"""
    # 初始化三个域的批处理列表
    domain_batches = [[], [], []]

    for sample_list in batch:
        for domain_idx, sample in enumerate(sample_list):
            domain_batches[domain_idx].append(sample)

    # 转换为Batch对象
    return [Batch.from_data_list(bd) for bd in domain_batches]


# # ################################################### 定义训练、测试函数 ################################################
# 训练函数
def train(model, train_loader, optimizer, scheduler, criterion, mmd_loss, dist_loss,
          device, num_epochs=70, lambda_d=1.0, lambda_mmd=1.0, lambda_dist=0.5, alpha=0.1):

    stats = {
        'DQ_loss': [], 'class_loss': [], 'domain_loss': [],
        'total_loss': [], 'class_acc': [], 'domain_acc': []
    }

    writer = SummaryWriter('runs/experiment_1')  # 创建一个 TensorBoard writer

    for epoch in range(num_epochs):
        print(f"Epoch [{epoch + 1}/{num_epochs}] - 开始训练")
        model.train()
        if epoch < 20:
            lambda_d = 0.0  # 暂时关闭域对抗，先让域分类器学习基本特征
        else:
            lambda_d = 5.0

        ### 损失累计变量初始化
        class_losses = 0.0
        DQ_total_losses = 0.0
        domain_losses = 0.0
        train_losses = 0.0

        ### 样本量累计变量初始化
        correct_class = 0
        correct_domain = 0
        total_samples = 0
        domain_total_samples = 0

        # 记录训练时间
        start_time = time.time()

        for i, (domain0_batch, domain1_batch, domain2_batch) in enumerate(train_loader):

            # 将数据移至设备
            domain0_batch = domain0_batch.to(device)
            domain1_batch = domain1_batch.to(device)
            domain2_batch = domain2_batch.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # 分别处理源域数据
            source0_feat, source0_class_out, source0_domain_out = model(domain0_batch)
            source1_feat, source1_class_out, source1_domain_out = model(domain1_batch)
            target_feat, _, target_domain_out = model(domain2_batch)

            # 计算MMD损失（源域与目标域的边缘对齐）
            mmd_loss1 = mmd_loss(source0_feat, target_feat)  # 源域1 → 目标域
            mmd_loss2 = mmd_loss(source1_feat, target_feat)  # 源域2 → 目标域
            # mmd_loss3 = mmd_loss(source0_feat, source1_feat) # 源域1 → 源域2  不需要的

            # 计算Dist损失（仅用于两个源域之间的条件对齐）
            dist_loss_value = dist_loss(source0_feat, domain0_batch.y, source1_feat, domain1_batch.y)

            # 1. 对齐总损失
            DQ_total_loss = lambda_mmd * (mmd_loss1 + mmd_loss2 )  + lambda_dist * dist_loss_value
            # DQ_total_loss = lambda_dist * dist_loss_value

            # 2. 源域分类总损失
            class_loss0 = criterion(source0_class_out, domain0_batch.y)
            class_loss1 = criterion(source1_class_out, domain1_batch.y)
            class_loss = (class_loss0 + class_loss1) / 2


            # 3. 自适应泛化总损失 (域对抗损失)
            # 源域1(标签0)(领域自适应)
            domain0_output = source0_domain_out
            domain0_target = torch.zeros(len(source0_feat), dtype=torch.long).to(device)
            domain0_loss = criterion(domain0_output, domain0_target)

            # 源域2(标签1)
            domain1_output = source1_domain_out
            domain1_target = torch.ones(len(source1_feat), dtype=torch.long).to(device)
            domain1_loss = criterion(domain1_output, domain1_target)

            # 目标域(标签2)
            target_domain_output = target_domain_out
            target_domain_target = torch.ones(len(target_feat), dtype=torch.long).to(device) * 2
            target_loss = criterion(target_domain_output, target_domain_target)

            domain_loss = (domain0_loss + domain1_loss + target_loss) / 3


            # 任务总损失,并进行反向传播(一个batch)
            total_loss = DQ_total_loss + class_loss + lambda_d * domain_loss
            # total_loss = class_loss + lambda_d * domain_loss
            total_loss.backward()
            optimizer.step()



            # 记录损失
            DQ_total_losses += DQ_total_loss.item()
            class_losses += class_loss.item()
            domain_losses += domain_loss.item()
            train_losses += total_loss.item()


            # 统计每个epoch源域正确分类（pre）和（inter）的样本总数（合并两个源域的结果）
            outputs1 = source0_class_out.detach()
            outputs2 = source1_class_out.detach()

            class_output = torch.cat([outputs1, outputs2], dim=0)
            source_labels = torch.cat([domain0_batch.y, domain1_batch.y], dim=0).detach()

            _, predicted = torch.max(class_output, 1)
            total_samples += source_labels.size(0)
            correct_class += (predicted == source_labels).sum().item()


            # 统计每个epoch源域正确分类（源域）和（目标域）的样本总数
            domain_outputs0 = source0_domain_out.detach()
            domain_outputs1 = source1_domain_out.detach()
            domain_target = target_domain_out.detach()

            domain_output = torch.cat([domain_outputs0,domain_outputs1,domain_target], dim=0)
            domain_labels = torch.cat([domain0_target,domain1_target,target_domain_target], dim=0).detach()

            _, domain_predicted = torch.max(domain_output, 1)
            domain_total_samples += domain_labels.size(0)
            correct_domain += (domain_predicted == domain_labels).sum().item()

        #  epoch 结束后统计
        epoch_time = time.time() - start_time

        # 计算对齐损失
        DQ_average_loss = DQ_total_losses / len(train_loader)

        # 计算总体损失
        total_average_loss = train_losses / len(train_loader)

        # 计算分类损失和准确率
        class_average_loss = class_losses / len(train_loader)
        class_accuracy = 100 * correct_class / total_samples

        # 计算域分类损失
        domain_average_loss = domain_losses / len(train_loader)
        domain_accuracy = 100 * correct_domain / domain_total_samples

        stats['DQ_loss'].append(DQ_average_loss)
        stats['class_loss'].append(class_average_loss)
        stats['domain_loss'].append(domain_average_loss)
        stats['total_loss'].append(total_average_loss)
        stats['class_acc'].append(class_accuracy)
        stats['domain_acc'].append(domain_accuracy)

        print(
            f'Epoch [{epoch + 1}/{num_epochs}]\n'
            f'时间: {epoch_time:.2f}s\n'
            f'DQ_loss: {DQ_average_loss:.4f}\n'
            f'class_loss: {class_average_loss:.2f}\n'
            f'class_accuracy: {class_accuracy:.2f}%\n'  # 建议添加百分比格式
            f'domain_loss: {domain_average_loss:.4f}\n'
            f'domain_accuracy: {domain_accuracy:.2f}%\n'  # 建议添加百分比格式
            f'total_loss: {total_average_loss:.4f}'  # 补充缺失的百分号或数值单位
        )


        # 将训练和验证的损失和准确率写入 TensorBoard
        writer.add_scalar('DQ_average_loss/train', DQ_average_loss, epoch)
        writer.add_scalar('total_average_loss/train', total_average_loss, epoch)
        writer.add_scalar('class_average_loss', class_average_loss, epoch)
        writer.add_scalar('domain_average_loss/train', domain_average_loss, epoch)
        writer.add_scalar('class_accuracy', class_accuracy, epoch)
        writer.add_scalar('domain_accuracy', domain_accuracy, epoch)
        # writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)

        # 更新学习率
        scheduler.step()


    print('Finished Training')
    writer.close()  # 关闭 TensorBoard writer
    return stats


def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():  # 关闭梯度计算，节省内存和计算资源
        for i, data in enumerate(test_loader):
            data = data.to(device)
            _, class_output, _ = model(data)

            # 分离张量，避免不必要的梯度计算
            class_outputs = class_output.detach()
            test_labels = data.y.detach()

            # 统计正确预测的样本数
            _, predicted = torch.max(class_outputs, 1)
            total += test_labels.size(0)
            correct += (predicted == test_labels).sum().item()

            # 收集所有标签和预测结果，用于后续分析
            all_labels.extend(test_labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    # 计算准确率
    accuracy = 100 * correct / total

    # 打印结果
    print(f'测试集准确率: {accuracy:.2f}%')

    # 计算其他评估指标（如精确率、召回率、F1分数）
    from sklearn.metrics import classification_report
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds, target_names=['Preictal', 'Interictal']))

    return accuracy, all_labels, all_preds

# ######################################################### 主程序 ########################################################
if __name__ == "__main__":
    # 设置随机种子
    set_seed(SEED)
    folder_path='runs'
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据路径（需根据实际调整）
    source1_path = "DE_MEAN/chb01"
    source2_path = "DE_MEAN/chb03"
    target_path = "DE_MEAN/chb05"

    # 固定邻接矩阵
    adj_list_sample = {
        0: [1, 4, 8, 9, 12, 13],
        1: [2, 8, 12],
        2: [3, 18],
        3: [7],
        4: [5, 8, 9, 12, 13],
        5: [6],
        6: [7],
        8: [9, 12],
        9: [10],
        10: [11],
        11: [15],
        12: [13],
        13: [14, 16],
        14: [15],
        16: [17, 5],
        17: [10],
        18: [19],
        19: [20],
        20: [14]
    }
    for i in range(21):
        if i not in adj_list_sample:
            adj_list_sample[i] = []
    num_nodes = 21
    adj_matrix = torch.tensor(build_adjacency_matrix(adj_list_sample, num_nodes), dtype=torch.float32)

    # 创建数据集
    dataset = MultiDomainDataset(source1_path, source2_path, target_path, adj_matrix)


    # 创建数据加载器
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    # 创建测试数据集（使用目标域的测试集）
    test_features, test_labels = dataset.target_test
    test_dataset = EEGTestDataset(test_features, test_labels, adj_matrix)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 创建模型
    model = EEGDANN(
        in_channels=IN_CHANNELS,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=OUT_CHANNELS,
        num_classes=NUM_CLASSES,
        num_nodes=NUM_NODES,
        device=device,
        fixed_adj_matrix=adj_matrix
    ).to(device)

    # 优化器
    optimizer = Adam(model.parameters(), lr=LR)

    # 定义MMD_loss、Dist_loss和分类损失函数
    criterion = nn.CrossEntropyLoss()
    mmd_loss = MMD_loss.MMD_loss()
    dist_loss = Dist_loss.Dist_Loss()

    # 学习率调度器
    scheduler = ExponentialLR(optimizer, gamma=GAMMA)

    # 调用训练和测试函数
    train(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        mmd_loss=mmd_loss,
        dist_loss=dist_loss,
        device=device,
        num_epochs=NUM_EPOCHS,
        lambda_d=LAMBDA_D,
        lambda_mmd=lambda_mmd,
        lambda_dist=lambda_dist,
        alpha=alpha
    )

    test(
        model=model,
        test_loader=test_loader,
        device=device
    )