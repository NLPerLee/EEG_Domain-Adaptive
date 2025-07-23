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
import utils
import math
from torch_geometric.data import Batch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import Parameter
import torch.nn as nn
import GRL
import os
import numpy as np
import torch
import random
from torch.utils.data import Dataset,random_split
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader as GeoDataLoader
import torch
import random
import numpy as np
import os
import time
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 设置随机种子,保证复现
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

# 根据邻接字典构建邻接矩阵（包含自连接）
def build_adjacency_matrix(adj_list, num_nodes):
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)  # 使用 float32 更适合后续 torch 和归一化操作
    for node, neighbors in adj_list.items():
        for neighbor in neighbors:
            adj_matrix[node, neighbor] = 1  # 原始边
            adj_matrix[neighbor, node] = 1  # 无向图双向连接

    # # 添加自连接（自己到自己的边）
    # np.fill_diagonal(adj_matrix, 1)

    return adj_matrix

# 使用SEnet为节点特征附加权重,突出重要特征,降低冗余特征对模型性能的影响
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 全局平均池化
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, n, c = x.size()  # (batch_size, num_nodes, feature_dim)
        y = self.avg_pool(x.permute(0, 2, 1)).view(b, c)  # Squeeze: (b, c)
        y = self.fc(y).view(b, c, 1)  # Excitation: (b, c, 1)
        return x * y.expand_as(x.permute(0, 2, 1)).permute(0, 2, 1)


# AdaptiveGCN图神经网络域(共享特征提取器)
class AdaptiveGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels_1, hidden_channels_2, out_channels,
                 fixed_adj_matrix=None, num_nodes=21, use_residual=True, device='cuda',se_reduction=4):
        """
        参数:
            in_channels: 输入特征维度
            hidden_channels_1: 第一层隐藏层维度
            hidden_channels_2: 第二层隐藏层维度
            out_channels: 输出特征维度
            fixed_adj_matrix: 固定邻接矩阵(可选)
            num_nodes: 节点数量
            use_residual: 是否使用残差连接
            device: 设备类型('cuda'或'cpu')
            se_reduction: SE模块中的降维系数
        """
        super(AdaptiveGCN, self).__init__()
        self.num_nodes = num_nodes
        self.device = device
        self.use_residual = use_residual

        # 初始图结构和自适应图结构融合权重,可学习
        self.weight = Parameter(torch.Tensor(1).normal_(mean=0, std=0.1))

        # 查看初始邻接矩阵
        self.fixed_adj_matrix = fixed_adj_matrix.to(device) if fixed_adj_matrix is not None else None

        # SE层，用于增强输入特征
        self.se_layer_input = SELayer(in_channels, se_reduction)

        # 图卷积层
        self.conv1 = GCNConv(in_channels, hidden_channels_1)
        self.conv2 = GCNConv(hidden_channels_1, hidden_channels_2)
        self.conv3 = GCNConv(hidden_channels_2, out_channels)

        # 批归一化层
        self.bn1 = nn.BatchNorm1d(hidden_channels_1)
        self.bn2 = nn.BatchNorm1d(hidden_channels_2)
        self.bn3 = nn.BatchNorm1d(out_channels)

        # 残差连接所需的线性映射（当维度不匹配时）
        self.res_skip1 = None
        self.res_skip2 = None

        if use_residual and in_channels != hidden_channels_1:
            self.res_skip1 = torch.nn.Linear(in_channels, hidden_channels_1)

        if use_residual and hidden_channels_1 != hidden_channels_2:
            self.res_skip2 = torch.nn.Linear(hidden_channels_1, hidden_channels_2)

        # 线性映射
        self.fc = torch.nn.Linear(out_channels, 2 * out_channels)


        # 自适应图结构的构建
        if fixed_adj_matrix is not None:
            # 确保 fixed_adj_matrix 是张量
            if not isinstance(fixed_adj_matrix, torch.Tensor):
                fixed_adj_matrix = torch.tensor(fixed_adj_matrix, dtype=torch.float32, device=device)
            # 使用SVD初始化
            m, p, n = torch.svd(fixed_adj_matrix)
            rank = min(num_nodes, 10)  # 设置rank为较小的值，比如10
            initemb1 = torch.mm(m[:, :rank], torch.diag(p[:rank] ** 0.5))
            initemb2 = torch.mm(torch.diag(p[:rank] ** 0.5), n[:, :rank].t())
            self.nodevec1 = Parameter(initemb1.to(device), requires_grad=True)
            self.nodevec2 = Parameter(initemb2.to(device), requires_grad=True)
        else:
            # 随机初始化
            self.nodevec1 = Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True)
            self.nodevec2 = Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True)

        # 保存训练后的结构，这样优化后的邻接矩阵可以和边权重可以和图神经网络中神经元的参数一起被存储,但不是模型参数
        self.register_buffer('learned_edge_index', None)
        self.register_buffer('learned_edge_weight', None)

    def symmetric_normalize_adj(self,adj):
        # 对称归一化邻接矩阵
        # 添加自连接
        adj_tilde = adj + torch.eye(adj.size(0), device=adj.device)
        # 计算度矩阵
        degree_matrix = torch.sum(adj_tilde, dim=1)
        # 计算度矩阵的逆平方根
        degree_matrix_inv_sqrt = degree_matrix.pow(-0.5)
        degree_matrix_inv_sqrt[degree_matrix_inv_sqrt == float('inf')] = 0
        # 构建对角矩阵
        D_inv_sqrt = torch.diag(degree_matrix_inv_sqrt)
        # 对称归一化
        normalized_adj = torch.mm(torch.mm(D_inv_sqrt, adj_tilde), D_inv_sqrt)
        return normalized_adj


    def forward(self, data):
        x = data.x
        batch = data.batch if hasattr(data, 'batch') else None

        # 在输入GNN前先通过SENet增强特征
        x = x.view(-1, self.num_nodes, x.size(1))  # Reshape for SE layer
        x = self.se_layer_input(x)
        x = x.view(-1, x.size(2))  # Reshape back to original shape

        # # # 获取批量大小和节点数
        # batch_size = batch.max().item() + 1 if batch is not None else 1
        # print(batch_size)
        # channels = 21
        # features = 5
        #
        # # Reshape x to (batch_size, channels, features)
        # x = x.view(-1, channels, features)
        # print(f"Reshaped x: {x.shape}")  # Debugging output
        #
        # # 确保 x 是连续的
        # x = x.contiguous()
        #
        # # 重塑输入为 (batch_size * num_nodes, in_channels)
        # x = x.view(-1, features)
        # x_mapped = F.relu(self.feature_mapper(x))

        # 训练阶段：生成并保存自适应结构
        if self.training:

            # 构建自适应邻接矩阵
            adp = torch.sigmoid(torch.mm(self.nodevec1, self.nodevec2))
            alpha = torch.sigmoid(self.weight)

            if self.fixed_adj_matrix is not None:
                # 将自适应邻接矩阵和固定邻接矩阵加权融合
                combined_adp = alpha * adp + (1 - alpha) * self.fixed_adj_matrix
            else:
                combined_adp = adp

            # 对称归一化（保持无向图性质）
            normalized_adp = self.symmetric_normalize_adj(combined_adp)

            # 保存学习到的结构(每次训练都更新),并使用阈值过滤以减少边数量
            threshold = 0.1
            mask = normalized_adp > threshold
            adaptive_edge_index = mask.nonzero().t().contiguous()
            edge_weight = normalized_adp[adaptive_edge_index[0], adaptive_edge_index[1]]
            self.learned_edge_index = adaptive_edge_index
            self.learned_edge_weight = edge_weight

        # 测试阶段：使用保存的结构
        else:
            # 确保测试时有可用的结构
            if self.learned_edge_index is None:
                raise RuntimeError("在测试前需要先进行训练以生成图结构")

            adaptive_edge_index = self.learned_edge_index
            edge_weight = self.learned_edge_weight

        # 第一层GCN + 残差连接
        identity1 = x
        x = self.conv1(x, adaptive_edge_index, edge_weight)
        x = self.bn1(x)
        if self.use_residual:
            if self.res_skip1 is not None:
                identity1 = self.res_skip1(identity1)
            x += identity1  # 残差连接
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # 第二层GCN + 残差连接
        identity2 = x
        x = self.conv2(x, adaptive_edge_index, edge_weight)
        x = self.bn2(x)
        if self.use_residual:
            if self.res_skip2 is not None:
                identity2 = self.res_skip2(identity2)
            x += identity2  # 残差连接
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # 第三层GCN
        x = self.conv3(x, adaptive_edge_index, edge_weight)
        x = self.bn3(x)

        # 全局池化
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long).to(x.device)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)

        # 图特征再次映射
        x = self.fc(x)

        return x

    def reset_learned_structure(self):
        """重置学习到的结构，用于重新训练"""
        self.learned_edge_index = None
        self.learned_edge_weight = None

# 不同域数据(特定特征提取器)
class DSFE(nn.Module):
    def __init__(self, in_features=128, out_features=128, hidden_features=64):
        super(DSFE, self).__init__()
        # 第一个全连接层
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.bn1 = nn.BatchNorm1d(hidden_features)

        # 第二个全连接层
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.bn2 = nn.BatchNorm1d(hidden_features)

        # 第三个全连接层
        self.fc3 = nn.Linear(hidden_features, out_features)
        self.bn3 = nn.BatchNorm1d(out_features)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.fc3(out)
        out = self.bn3(out)

        return out


# 整体网络结构
class GMSDA(nn.Module):
    def __init__(self, number_of_source=2,num_classes=2,fix_matrix=None):
        super(GMSDA, self).__init__()

        # 共享特征提取器
        self.sharenet = AdaptiveGCN(in_channels=48, hidden_channels_1=256, hidden_channels_2=128, out_channels=64,
                 fixed_adj_matrix = fix_matrix, num_nodes=21, use_residual=True, device='cuda')

        # 使用 ModuleList 按顺序存储 DSFE 模块
        self.dsfe_modules = nn.ModuleList()
        # 使用 ModuleList 按顺序存储分类器模块
        self.cls_fc_modules = nn.ModuleList()

        for i in range(number_of_source):
            # 按顺序添加 DSFE 模块，索引 i 对应第 i 个源域
            self.dsfe_modules.append(DSFE(in_features=128, out_features=128, hidden_features=64))
            # 按顺序添加分类器，与 DSFE一一对应
            self.cls_fc_modules.append(nn.Linear(128, num_classes))


    def forward(self,data_source,data_tgt,mark=0):
        """
        :param data_source:单个源域
        :param data_tgt: 目标域数据
        :param mark: 分支标签
        :mmd_loss:目标域和源域之间的损失
        :disc_loss: 不同分支下目标域数据的差异
        :cls_loss:源域的分类损失
        :source_align_loss: 源域间对齐损失
        """
        mmd_loss = 0
        disc_loss = 0
        data_tgt_dsfe = []
        data_tgt_cls = []
        number_of_source = len(self.dsfe_modules)  # 从内部获取源域数量

        if self.training == True:
            # 源域和目标域在公共特征提取器的特征
            source_data = data_source.to(device)
            target_data = data_tgt.to(device)
            data_src_share = self.sharenet(source_data)
            data_tgt_share = self.sharenet(target_data)

            # # 假设 data_source 和 data_tgt 已经被预处理成相同形状的张量,拼接的方式能够减少计算开销,因为减少了数据加载
            # source_feature = data_source.x.to(device)
            # target_feature = data_tgt.x.to(device)
            # combined_data = torch.cat([source_feature, target_feature], dim=0)  # 在第0维度上拼接
            # shared_features = self.sharenet(combined_data)
            # # 根据数据集大小切分回各自的特征表示
            # data_src_share = shared_features[:data_source.size(0)]
            # data_tgt_share = shared_features[data_source.size(0):]

            # 得到每个分支下目标域的特征表示
            for i in range(number_of_source):
                # 直接通过ModuleList索引调用第i个DSFE
                tgt_fea = self.dsfe_modules[i](data_tgt_share)
                tgt_cls = self.cls_fc_modules[i](tgt_fea)
                data_tgt_dsfe.append(tgt_fea)
                data_tgt_cls.append(tgt_cls)

            # 得到源域在当前分支下的特征表示
            data_src_dsfe = self.dsfe_modules[mark](data_src_share)

            # 计算对应分支下源域和目标域之间的分布
            mmd_loss += utils.mmd_linear(data_src_dsfe,data_tgt_dsfe[mark])

            # 计算每个分支下目标域之间的距离，距离越小越好
            for i in range(number_of_source):
                if i != mark:
                    disc_loss += torch.mean(torch.abs(
                        F.softmax(data_tgt_cls[mark], dim=1) -
                        F.softmax(data_tgt_cls[i], dim=1)
                    ))

            # 计算源域的分类损失
            label_src = data_source.y.to(device)
            pred_src = self.cls_fc_modules[mark](data_src_dsfe)  # 当前分支分类器预测
            cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src.squeeze())

            return cls_loss, mmd_loss, disc_loss

        else:
            # 测试模式：输入数据（通常为目标域）经过所有分支，返回所有预测结果
            data_share = self.sharenet(data_tgt)  # 共享特征提取
            preds = []
            for i in range(number_of_source):
                # 每个分支的域特定特征提取 + 分类
                fea = self.dsfe_modules[i](data_share)
                pred = self.cls_fc_modules[i](fea)
                preds.append(pred)
            return preds


# 优化后的训练函数
def train(
        model,
        source_loaders,
        target_loader,
        device,
        lr=0.001,
        max_iterations=10000,
        log_interval=10,
        test_fn=None,
        test_loader=None,
        test_interval=200,
        save_best=True,
        checkpoint_path='best_model.pth'
):
    """
    训练GMSDA模型

    参数:
        model: GMSDA模型实例
        source_loaders: 源域数据加载器列表
        target_loader: 目标域数据加载器
        device: 计算设备
        lr: 学习率
        max_iterations: 最大迭代次数
        log_interval: 日志打印间隔
        test_fn: 测试函数
        test_loader: 测试数据加载器
        test_interval: 测试间隔
        save_best: 是否保存最佳模型
        checkpoint_path: 最佳模型保存路径
    """

    # 创建源域和目标域的迭代器
    source_iters = [iter(loader) for loader in source_loaders]
    target_iter = iter(target_loader)

    best_accuracy = 0.0

    for i in range(1, max_iterations + 1):
        # 动态调整学习率
        model.train()
        model.to(device)
        # lr_rate = lr / math.pow((1 + 10 * (i - 1) / (max_iterations)), 0.75)
        lr_rate = lr * (0.5 ** (i // 2000))
        # lr_rate = lr

        # 优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)

        # 遍历每个源域分支
        for j in range(len(source_loaders)):
            # 获取源域和目标域的batch数据
            try:
                source_batch = next(source_iters[j])
            except StopIteration:
                source_iters[j] = iter(source_loaders[j])
                source_batch = next(source_iters[j])

            try:
                target_batch = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_batch = next(target_iter)

            # 将数据移至设备
            source_data = source_batch
            target_data = target_batch

            # 前向传播
            optimizer.zero_grad()
            cls_loss, mmd_loss, disc_loss = model(
                data_source=source_data,
                data_tgt=target_data,
                mark=j
            )

            # 计算动态权重
            gamma = 2 / (1 + math.exp(-10 * (i) / (max_iterations))) - 1
            beta = gamma / 100

            # 总损失
            loss = cls_loss + gamma * mmd_loss + beta * disc_loss

            # 反向传播
            loss.backward()
            optimizer.step()

            # 打印训练日志
            if i % log_interval == 0:
                print(f'Train iter: {i} [{100. * i / max_iterations:.0f}%]\t'
                      f'Loss: {loss.item():.6f}\tcls_Loss: {cls_loss.item():.6f}\t'
                      f'mmd_Loss: {mmd_loss.item():.6f}\tdisc_Loss: {disc_loss.item():.6f}')

        # 定期测试模型
        if test_fn and test_loader and i % test_interval == 0:
            model.eval()
            with torch.no_grad():
                correct = test_fn(model, test_loader, device)
                accuracy = 100. * correct / len(test_loader.dataset)
                print(f'Test accuracy at iter {i}: {accuracy:.2f}%')

                # 保存最佳模型
                if save_best and accuracy > best_accuracy:
                    best_accuracy = accuracy
                    torch.save(model.state_dict(), checkpoint_path)
                    print(f'Best model saved with accuracy: {best_accuracy:.2f}%')

            model.train()

    return best_accuracy if save_best else None


def test(model, test_loader, device):
    """
    测试GMSDA模型

    参数:
        model: GMSDA模型实例
        data_loader: 测试数据加载器
        device: 计算设备

    返回:
        正确预测的样本数和各分支准确率列表
    """
    model.eval()
    model.to(device)

    test_loss = 0
    correct = 0
    branch_corrects = [0] * len(model.dsfe_modules)  # 每个分支的正确预测数

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            # 获取数据并移至设备
            data = data.to(device)

            # 模型前向传播（测试模式）
            preds = model(data_tgt=data)  # 所有分支的预测结果

            # 对每个分支的预测应用softmax
            softmax_preds = [F.softmax(pred, dim=1) for pred in preds]

            # 集成所有分支的预测结果（平均）
            ensemble_pred = sum(softmax_preds) / len(softmax_preds)

            # 计算集成损失
            test_label = data.y.to(device)
            test_loss += F.nll_loss(
                F.log_softmax(ensemble_pred, dim=1),
                test_label.squeeze()
            ).item()

            # 计算集成预测的准确率
            pred_labels = ensemble_pred.argmax(dim=1)  # 获取预测的类别标签
            correct += pred_labels.eq(test_label.squeeze()).sum().item()

            # 计算每个分支单独的准确率
            for branch_idx in range(len(softmax_preds)):
                branch_pred = softmax_preds[branch_idx].argmax(dim=1)
                branch_corrects[branch_idx] += branch_pred.eq(test_label.squeeze()).sum().item()

    # 计算平均损失
    test_loss /= len(test_loader.dataset)

    # 计算各分支的准确率
    branch_accuracies = [100.0 * c / len(test_loader.dataset) for c in branch_corrects]

    # 打印测试结果
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Ensemble Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100.0 * correct / len(test_loader.dataset):.2f}%)\n')

    # 打印每个分支的准确率
    for i, acc in enumerate(branch_accuracies):
        print(f'Branch {i} Accuracy: {acc:.2f}%')

    return correct


### 加载数据集,并且处理eeg数据
def load_and_process_eeg_data(folder_path, label, normalize=True):
    data_list = []
    labels_list = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.npy'):
            file_path = os.path.join(folder_path, filename)
            eeg_segment = np.load(file_path)  # 假设形状为 (21, 48)

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

    return features, labels

### 加载为eeg数据集
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



if __name__ == "__main__":
    # 使用您选择的种子数
    set_seed(42)

    # 确定模型加载到GPU上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    num_nodes = 21
    adj_matrix = torch.tensor(build_adjacency_matrix(adj_list_sample, num_nodes), dtype=torch.float32)


    # 初始化模型
    model = GMSDA(number_of_source=2, num_classes=2,fix_matrix=adj_matrix)

    # 数据路径
    source1_path = "E:\Seizure_Types\\DE_MEAN\chb01"
    source2_path = "E:\Seizure_Types\\DE_MEAN\chb03"
    target_path = "E:\Seizure_Types\\DE_MEAN\chb05"

    # 加载两个源域数据
    source1_features, source1_labels = load_domain_data(source1_path)
    source2_features, source2_labels = load_domain_data(source2_path)
    ds_source1 = EEGTestDataset(source1_features, source1_labels,adj_matrix)
    ds_source2 = EEGTestDataset(source2_features, source2_labels, adj_matrix)

    # 加载目标域数据
    target_features, target_labels = load_domain_data(target_path)

    # 将目标域数据划分为80%训练 + 20%测试
    ds_target = EEGTestDataset(target_features, target_labels, adj_matrix)
    train_size = int(0.8 * len(ds_target))
    test_size = len(ds_target) - train_size
    target_train_dataset, target_test_dataset = random_split(ds_target, [train_size, test_size])

    # 构建Dataloader
    batch_size = 32

    source1_loader = DataLoader(ds_source1, batch_size=batch_size, shuffle=True)
    source2_loader = DataLoader(ds_source2, batch_size=batch_size, shuffle=True)
    target_train_loader = DataLoader(target_train_dataset, batch_size=batch_size, shuffle=True)
    target_test_loader = DataLoader(target_test_dataset, batch_size=batch_size, shuffle=False)

    source_loaders = [source1_loader, source2_loader]  # 两个源域的DataLoader


    # 训练模型
    best_accuracy = train(
        model=model,
        source_loaders=source_loaders,
        target_loader=target_train_loader,
        device=device,
        lr=0.001,
        max_iterations=10000,
        log_interval=10,
        test_fn=test,
        test_loader=target_test_loader,
        test_interval=200,
        save_best=True,
        checkpoint_path='gmsda_best_model.pth'
    )

    print(f"Training completed. Best accuracy: {best_accuracy:.2f}%")

    # 加载最佳模型并测试
    model.load_state_dict(torch.load('gmsda_best_model.pth'))
    correct = test(model, target_test_loader, device)
    final_accuracy = 100.0 * correct / len(target_test_loader.dataset)
    print(f"Final test accuracy: {final_accuracy:.2f}%")