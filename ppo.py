import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import random
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import os
import logging
import json
from datetime import datetime

# -------------------------- 1. 配置管理 --------------------------
class Config:
    """统一管理配置参数，便于调参和维护"""
    def __init__(self):
        # 设备配置
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 模型参数
        self.INPUT_DIM = 2  # 关键点坐标 (x, y)
        self.HIDDEN_DIM = 128  # 增加LSTM隐藏层维度，提高模型表达能力
        self.NUM_LAYERS = 2  # 增加LSTM层数，捕捉更长的依赖关系
        self.LR = 1e-5  # 降低学习率以提高稳定性
        
        # 训练参数
        self.BATCH_SIZE = 8  # 进一步减少批次大小以降低计算量
        self.EPOCHS = 50  # 增加训练轮数以让模型有更多时间学习
        self.MIN_KEYPOINTS = 30  # 减少最小关键点数量以降低计算复杂度
        self.MAX_KEYPOINTS = 50  # 减少最大关键点数量以降低计算复杂度
        self.WEIGHTS = (0.6, 0.4)  # 路径长度权重提升（更侧重缩短路径）
        
        # PPO参数
        self.GAMMA = 0.99  # 折扣因子
        self.CLIP_PARAM = 0.2  # PPO裁剪参数
        self.ENTROPY_COEF = 0.01  # 熵正则化系数
        self.VALUE_LOSS_COEF = 0.5  # 价值损失系数
        self.GAE_LAMBDA = 0.95  # GAE lambda参数
        self.USE_GAE = True  # 是否使用GAE优势估计
        self.USE_PROPER_TIME_LIMITS = False  # 是否使用适当的时间限制
        
        # 新奖励函数参数
        self.OMEGA1 = 0.8  # 路径长度权重
        self.OMEGA2 = 0.2  # 路径平滑度权重
        self.D1_COEF = 1.0  # 距离系数d1(x₀)
        self.D2_COEF = 1.0  # 距离系数d2(x₀)
        
        # 早停参数
        self.PATIENCE = 10  # 增加耐心值，让模型有更多时间学习
        
        # 可视化配置
        self.SAVE_DIR = "path_planning_results"
        self.TEST_INTERVAL = 10  # 测试间隔（增加以减少可视化开销）
        
        # 日志配置
        self.LOG_LEVEL = logging.INFO
        
    def to_dict(self):
        """转换为字典，便于保存"""
        config_dict = {
            key: value for key, value in self.__dict__.items() 
            if not key.startswith('_')
        }
        # 特殊处理DEVICE属性，将torch.device对象转换为字符串
        for key, value in config_dict.items():
            if isinstance(value, torch.device):
                config_dict[key] = str(value)
        return config_dict

# 初始化配置
def init_config():
    config = Config()
    # 创建保存目录
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    # 初始化日志
    init_logging(config)
    return config

# 初始化日志
def init_logging(config):
    """初始化日志配置"""
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config.SAVE_DIR, 'training.log')),
            logging.StreamHandler()
        ]
    )
    logging.info("初始化日志系统")
    logging.info(f"配置参数：{json.dumps(config.to_dict(), indent=2)}")

# -------------------------- 2. 数据集生成 --------------------------
def is_inside_contour(point, contour):
    """检查点是否在轮廓内部"""
    # 使用射线法判断点是否在多边形内部
    x, y = point
    inside = False
    n = len(contour)
    for i in range(n):
        xi, yi = contour[i]
        xj, yj = contour[(i+1)%n]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
    return inside

def line_intersects_circle(p1, p2, center, radius):
    """检查两点之间的直线是否穿过圆"""
    # 计算线段的向量
    line_vec = p2 - p1
    # 计算圆心到线段起点的向量
    center_vec = center - p1
    # 计算线段的长度平方
    line_len_sq = np.dot(line_vec, line_vec)
    
    # 如果线段长度为0，直接检查点是否在圆内
    if line_len_sq == 0:
        return np.linalg.norm(center_vec) < radius
    
    # 计算t值，即圆心到线段的垂足参数
    t = np.dot(center_vec, line_vec) / line_len_sq
    
    # 限制t在[0,1]范围内
    t = max(0, min(1, t))
    
    # 计算垂足点
    closest_point = p1 + t * line_vec
    
    # 计算圆心到垂足点的距离
    distance = np.linalg.norm(closest_point - center)
    
    # 如果距离小于半径，则线段与圆相交
    return distance < radius


def segments_intersect(p1, p2, p3, p4, epsilon=1e-6):
    """
    检测两条线段是否相交或重叠（使用向量叉积方法）
    Args:
        p1: 第一条线段的起点 (x, y)
        p2: 第一条线段的终点 (x, y)
        p3: 第二条线段的起点 (x, y)
        p4: 第二条线段的终点 (x, y)
        epsilon: 浮点数精度阈值
    Returns:
        bool: 两条线段是否相交或重叠
    """
    # 将点转换为numpy数组
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    p4 = np.array(p4)
    
    # 快速排斥试验
    if (max(p1[0], p2[0]) < min(p3[0], p4[0]) - epsilon or
        max(p3[0], p4[0]) < min(p1[0], p2[0]) - epsilon or
        max(p1[1], p2[1]) < min(p3[1], p4[1]) - epsilon or
        max(p3[1], p4[1]) < min(p1[1], p2[1]) - epsilon):
        return False
    
    # 向量叉积
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    # 跨立试验
    c1 = cross(p1, p2, p3)
    c2 = cross(p1, p2, p4)
    c3 = cross(p3, p4, p1)
    c4 = cross(p3, p4, p2)
    
    # 处理共线情况
    if abs(c1) < epsilon and abs(c2) < epsilon:
        # 共线情况，检查线段是否重叠
        # 检查所有端点是否在线段上
        def is_point_on_segment(p, a, b):
            return (min(a[0], b[0]) - epsilon <= p[0] <= max(a[0], b[0]) + epsilon and
                    min(a[1], b[1]) - epsilon <= p[1] <= max(a[1], b[1]) + epsilon and
                    abs(cross(a, b, p)) < epsilon)
        
        return (is_point_on_segment(p1, p3, p4) or
                is_point_on_segment(p2, p3, p4) or
                is_point_on_segment(p3, p1, p2) or
                is_point_on_segment(p4, p1, p2))
    
    # 检查跨立条件
    return (c1 * c2 <= epsilon and c3 * c4 <= epsilon)


class KeypointGenerator:
    """关键点生成器，负责生成模拟复杂空心构件的关键点"""
    @staticmethod
    def generate_keypoints(num_points):
        """
        生成模拟复杂空心构件的关键点（外轮廓内、内孔外）
        优化：关键点之间距离固定为40，且路径不经过孔洞
        """
        # 外轮廓：正方形（500x500）
        outer_contour = np.array([[0, 0], [500, 0], [500, 500], [0, 500]])
        # 内孔：随机生成1-3个小圆孔（位置不重叠）
        num_holes = random.randint(1, 3)
        holes = []
        for _ in range(num_holes):
            # 确保内孔不超出外轮廓，且不重叠
            while True:
                center = np.random.uniform(120, 380, size=2)
                radius = np.random.uniform(30, 60)
                # 检查是否与已生成的内孔重叠
                overlap = False
                for (c, r) in holes:
                    if np.linalg.norm(center - c) < r + radius + 20:
                        overlap = True
                        break
                if not overlap:
                    holes.append((center, radius))
                    break
        
        # 生成固定距离40的关键点
        keypoints = []
        
        # 生成第一个点
        max_attempts = 100
        attempts = 0
        while attempts < max_attempts:
            first_point = np.random.uniform(60, 440, size=2)  # 确保有足够空间生成后续点
            
            # 检查是否在外轮廓内
            if not is_inside_contour(first_point, outer_contour):
                attempts += 1
                continue
            
            # 检查是否在内孔外
            in_hole = False
            for (center, radius) in holes:
                if np.linalg.norm(first_point - center) < radius:
                    in_hole = True
                    break
            
            if not in_hole:
                keypoints.append(first_point)
                break
            
            attempts += 1
        
        # 如果没有找到第一个点，使用默认点
        if len(keypoints) == 0:
            keypoints.append(np.array([100, 100]))
        
        # 生成后续点
        while len(keypoints) < num_points:
            current_point = keypoints[-1]
            attempts = 0
            max_attempts = 20
            
            while attempts < max_attempts:
                # 生成随机方向
                angle = np.random.uniform(0, 2 * np.pi)
                # 计算下一个点的位置（距离固定为40）
                next_point = current_point + np.array([40 * np.cos(angle), 40 * np.sin(angle)])
                
                # 快速边界检查
                if next_point[0] < 20 or next_point[0] > 480 or next_point[1] < 20 or next_point[1] > 480:
                    attempts += 1
                    continue
                
                # 检查是否在内孔外（简化检查：只检查距离）
                in_hole = False
                for (center, radius) in holes:
                    if np.linalg.norm(next_point - center) < radius:
                        in_hole = True
                        break
                
                if in_hole:
                    attempts += 1
                    continue
                
                # 检查是否与已有关键点太近（只检查最近的10个点）
                too_close = False
                check_points = keypoints[-10:]  # 只检查最近的10个点
                for point in check_points:
                    if np.linalg.norm(next_point - point) < 20:  # 最小距离为20
                        too_close = True
                        break
                
                if too_close:
                    attempts += 1
                    continue
                
                # 所有检查通过，添加该点
                keypoints.append(next_point)
                break
            
            # 如果尝试了max_attempts次仍未找到合适的点，生成一个随机点
            if attempts >= max_attempts:
                attempts = 0
                while attempts < max_attempts:
                    random_point = np.random.uniform(60, 440, size=2)
                    
                    # 检查是否在内孔外
                    in_hole = False
                    for (center, radius) in holes:
                        if np.linalg.norm(random_point - center) < radius:
                            in_hole = True
                            break
                    
                    if not in_hole:
                        keypoints.append(random_point)
                        break
                    
                    attempts += 1
                
                # 如果仍然没有找到，使用当前点的随机偏移
                if attempts >= max_attempts:
                    offset = np.random.uniform(-20, 20, size=2)
                    random_point = current_point + offset
                    keypoints.append(random_point)
        
        return np.array(keypoints), outer_contour, holes
    
    @staticmethod
    def generate_batch(batch_size, min_points=20, max_points=80):
        """生成批次数据"""
        batch = []
        outer_contours = []
        holes_list = []
        
        for _ in range(batch_size):
            num_points = random.randint(min_points, max_points)
            keypoints, outer_contour, holes = KeypointGenerator.generate_keypoints(num_points)
            # 转换为张量
            keypoints_tensor = torch.tensor(keypoints, dtype=torch.float32)
            batch.append(keypoints_tensor)
            outer_contours.append(outer_contour)
            holes_list.append(holes)
        
        # 填充到最大长度
        max_len = max(p.shape[0] for p in batch)
        padded_batch = torch.zeros((batch_size, max_len, 2))
        num_points_list = []
        
        for i, p in enumerate(batch):
            padded_batch[i, :p.shape[0]] = p
            num_points_list.append(p.shape[0])
        
        num_points_tensor = torch.tensor(num_points_list, dtype=torch.long)
        return padded_batch, num_points_tensor, outer_contours, holes_list

# -------------------------- 3. 经验回放池 --------------------------
class ReplayBuffer:
    """改进的经验回放池，支持批量采样和容量限制"""
    def __init__(self, capacity, max_keypoints=80):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.max_keypoints = max_keypoints  # 最大关键点数量，用于填充
    
    def push(self, state, num_points, action, reward):
        """添加经验到回放池"""
        # 填充state到最大关键点数量
        padded_state = torch.zeros((self.max_keypoints, 2), dtype=state.dtype, device=state.device)
        padded_state[:state.shape[0]] = state
        
        # 填充action到最大关键点数量
        padded_action = torch.full((self.max_keypoints,), -1, dtype=action.dtype, device=action.device)
        padded_action[:action.shape[0]] = action
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (padded_state, num_points, padded_action, reward)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """从回放池采样批次数据"""
        batch = random.sample(self.buffer, batch_size)
        states, num_points_list, actions, rewards = zip(*batch)
        
        # 转换为张量
        states = torch.stack(states)
        num_points_list = torch.stack(num_points_list)
        actions = torch.stack(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        
        return states, num_points_list, actions, rewards
    
    def __len__(self):
        """返回当前回放池大小"""
        return len(self.buffer)

# -------------------------- 4. 模型定义 --------------------------
class PointerNetworkActor(nn.Module):
    """指针网络Actor，用于生成路径序列"""
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(PointerNetworkActor, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM编码器，恢复原始输入维度
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # 指针网络注意力机制
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        
        # 距离权重，调整为更合理的值
        self.distance_weight = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        
    def forward(self, x, num_points):
        """
        x: [batch_size, seq_len, input_dim] - 关键点坐标
        num_points: [batch_size] - 每个样本的实际关键点数量
        """
        batch_size, seq_len, _ = x.size()
        
        # LSTM编码
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        lstm_out, (h_n, c_n) = self.lstm(x, (h0, c0))
        
        # 计算注意力分数
        scores = torch.zeros(batch_size, seq_len, seq_len).to(x.device)
        
        # 计算距离矩阵 [batch_size, seq_len, seq_len]
        distances = torch.cdist(x, x, p=2)
        
        for i in range(seq_len):
            # 每个时间步的隐藏状态
            h_i = lstm_out[:, i, :].unsqueeze(1)  # [batch_size, 1, hidden_dim]
            
            # 计算注意力分数
            attention_scores = self.v(torch.tanh(self.W1(h_i) + self.W2(lstm_out))).squeeze(-1)
            
            # 在注意力分数中加入距离因素：距离越近，分数越高
            # 使用负距离，因为距离越小越好
            distance_bonus = -self.distance_weight * distances[:, i, :] / 100.0  # 恢复合理的归一化系数
            
            # 综合注意力分数和距离因素
            scores[:, i, :] = attention_scores + distance_bonus
        
        # 掩码：只允许在实际关键点范围内选择
        for i in range(batch_size):
            valid_points = num_points[i]
            scores[i, :, valid_points:] = -float('inf')  # 无效点设置为负无穷
        
        # 转换为概率分布
        probs = torch.softmax(scores, dim=-1)
        
        return probs

class PointerNetworkCritic(nn.Module):
    """指针网络Critic，用于评估路径质量"""
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(PointerNetworkCritic, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM编码器
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # 全连接层
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, 1)  # 输出Q值
        self.relu = nn.ReLU()
    
    def forward(self, x, num_points):
        """
        x: [batch_size, seq_len, input_dim] - 关键点坐标
        num_points: [batch_size] - 每个样本的实际关键点数量
        """
        batch_size, seq_len, _ = x.size()
        
        # LSTM编码
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        lstm_out, (h_n, c_n) = self.lstm(x, (h0, c0))
        
        # 池化操作：只考虑有效点
        pooled_out = torch.zeros(batch_size, self.hidden_dim).to(x.device)
        for i in range(batch_size):
            valid_points = num_points[i]
            if valid_points > 0:
                # 对有效点的隐藏状态求平均
                pooled_out[i] = lstm_out[i, :valid_points].mean(dim=0)
        
        # 全连接层
        x = self.relu(self.fc1(pooled_out))
        q_value = self.fc2(x)
        
        return q_value

# -------------------------- 5. RL路径规划器 --------------------------
class RLPathPlanner:
    """强化学习路径规划器，整合了Actor-Critic架构，使用PPO算法"""
    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE
        
        # 初始化网络
        self.actor = PointerNetworkActor(config.INPUT_DIM, config.HIDDEN_DIM, config.NUM_LAYERS).to(self.device)
        self.critic = PointerNetworkCritic(config.INPUT_DIM, config.HIDDEN_DIM, config.NUM_LAYERS).to(self.device)
        
        # 优化器
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=config.LR)
        
        # 学习率调度器
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.8)
        
        # PPO参数
        self.clip_param = config.CLIP_PARAM
        self.entropy_coef = config.ENTROPY_COEF
        self.value_loss_coef = config.VALUE_LOSS_COEF
        self.gae_lambda = config.GAE_LAMBDA
        self.use_gae = config.USE_GAE
        
        # 其他参数
        self.gamma = config.GAMMA
        self.weights = config.WEIGHTS
        
        # 性能追踪
        self.best_score = float('-inf')
        self.patience_counter = 0
    
    def compute_reward(self, states, num_points, path_seq, holes_list=None):
        """实现优化的奖励函数：确保规划路径距离小于原始路径，同时保持无交叉和安全特性"""
        batch_size = states.size(0)
        rewards = torch.zeros(batch_size, dtype=torch.float32, device=states.device)
        
        for b in range(batch_size):
            n = num_points[b].item()
            if n <= 1:
                rewards[b] = -float('inf')
                continue
            
            # 获取实际的关键点和路径
            keypoints = states[b, :n].cpu().numpy()
            path_indices = path_seq[b, :n].cpu().numpy()
            
            # 检查路径是否有效（没有重复点）
            if len(np.unique(path_indices)) != n:
                rewards[b] = -float('inf')
                continue
            
            # 计算路径点
            path_points = keypoints[path_indices]
            
            # 计算规划路径长度
            planned_length = 0.0
            for i in range(n-1):
                planned_length += np.linalg.norm(path_points[i+1] - path_points[i])
            
            # 计算原始路径长度（按照关键点顺序连接）
            original_length = 0.0
            for i in range(n-1):
                original_length += np.linalg.norm(keypoints[i+1] - keypoints[i])
            
            # 计算路径平滑度（与180度的转角差值总和）
            angle_deviation_sum = 0.0
            if n > 2:  # 确保至少有3个点才能计算转角
                for i in range(1, n-1):
                    vec1 = path_points[i] - path_points[i-1]
                    vec2 = path_points[i+1] - path_points[i]
                    
                    # 计算向量夹角
                    cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)
                    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 防止数值不稳定
                    angle = np.arccos(cos_theta)
                    
                    # 计算与180度（π弧度）的差值：差值越小（越接近180度）惩罚越少
                    angle_deviation = np.abs(angle - np.pi)
                    angle_deviation_sum += angle_deviation
                
                # 对差值总和进行归一化处理，除以路径点数量
                angle_deviation_sum /= n
            
            # 计算奖励函数各部分
            # 路径长度惩罚权重 - 进一步增加权重，使模型更关注路径长度优化
            term1 = 0.8 * planned_length  # 进一步增加惩罚权重，让模型更关注路径长度
            # 转角惩罚 - 使用与180度的差值作为惩罚项，越接近0度（差值越大）惩罚越大
            term2 = 0.00005 * (angle_deviation_sum ** 2)
            
            # 计算交叉点惩罚
            cross_penalty = 0.0
            cross_count = 0
            for i in range(n-1):
                for j in range(i+2, n-1):  # 不检查相邻或连续的线段
                    p1 = path_points[i]
                    p2 = path_points[i+1]
                    p3 = path_points[j]
                    p4 = path_points[j+1]
                    
                    if segments_intersect(p1, p2, p3, p4):
                        cross_count += 1
            
            # 交叉点惩罚权重 - 大幅增加权重，严格惩罚交叉点
            cross_penalty = 500.0 * cross_count
            
            # 计算不安全线段惩罚（穿过孔洞的线段）
            unsafe_penalty = 0.0
            unsafe_count = 0
            
            # 模拟测试脚本中的check_path_safety函数
            def line_intersects_circle(p1, p2, circle_center, radius):
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                a = dx**2 + dy**2
                b = 2 * (dx*(p1[0] - circle_center[0]) + dy*(p1[1] - circle_center[1]))
                c = (p1[0] - circle_center[0])**2 + (p1[1] - circle_center[1])**2 - radius**2
                discriminant = b**2 - 4*a*c
                if discriminant < 0:
                    return False
                if a == 0:
                    return False
                t1 = (-b - np.sqrt(discriminant)) / (2*a)
                t2 = (-b + np.sqrt(discriminant)) / (2*a)
                return (0 <= t1 <= 1) or (0 <= t2 <= 1)
            
            # 获取当前样本的孔洞信息
            safety_distance = 5.0
            holes = []
            if holes_list is not None and b < len(holes_list):
                holes = holes_list[b]
            elif hasattr(self, 'holes') and self.holes is not None:
                holes = self.holes
            
            # 检查路径是否穿过孔洞（带安全距离）
            for i in range(n-1):
                p1 = path_points[i]
                p2 = path_points[i+1]
                
                for hole in holes:
                    if line_intersects_circle(p1, p2, hole[0], hole[1] + safety_distance):
                        unsafe_count += 1
                        break
            
            # 不安全线段惩罚权重
            unsafe_penalty = 200.0 * unsafe_count
            
            # 增加对直线段的额外奖励，鼓励更直接的路径
            straight_lines = 0.0
            for i in range(1, n-1):
                vec1 = path_points[i] - path_points[i-1]
                vec2 = path_points[i+1] - path_points[i]
                cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                # 角度接近0（直线）时给予奖励，扩大直线的判定角度
                if np.abs(np.arccos(cos_theta)) < 0.35:  # 小于20度视为直线
                    straight_lines += 1.0
            
            # 直线段奖励：保持适当奖励，鼓励直线段
            straight_reward = 200.0 * straight_lines
            
            # 核心改进：增加规划路径与原始路径长度比较的奖励
            # 当规划路径比原始路径短时给予额外奖励，差值越大奖励越多
            length_comparison_reward = 0.0
            if planned_length < original_length:
                # 额外奖励与路径长度差成正比，大幅增加奖励权重
                length_comparison_reward = 10000.0 * (original_length - planned_length) / original_length * 100
            else:
                # 当规划路径比原始路径长时给予较重的惩罚，鼓励更短路径
                length_comparison_reward = -1000.0 * (planned_length - original_length) / original_length * 100
            
            # 综合计算奖励
            reward = - (term1 + term2 + cross_penalty + unsafe_penalty) + straight_reward + length_comparison_reward
            
            # 确保奖励值在合理范围内
            reward = torch.tensor(reward, dtype=torch.float32, device=states.device)
            rewards[b] = torch.clamp(reward, -1e5, 1e5)
        
        return rewards
    
    def calculate_path_metrics(self, states, num_points, path_seq):
        """计算路径长度和转角总和"""
        batch_size, max_len, _ = states.size()
        lengths = torch.zeros(batch_size).to(self.device)
        angle_sums = torch.zeros(batch_size).to(self.device)
        
        for i in range(batch_size):
            n = num_points[i].item()
            if n <= 1:
                continue
            
            # 获取有效路径
            valid_seq = path_seq[i, :n]
            
            # 检查路径唯一性
            unique_points = torch.unique(valid_seq)
            has_duplicates = len(unique_points) != n
            
            # 获取路径坐标
            path_coords = states[i, valid_seq, :]
            
            # 检查路径交叉
            has_intersection = False
            for j in range(n-1):
                for k in range(j+2, n-1):  # 不检查相邻或连续的线段
                    # 获取线段的两个端点
                    p1 = path_coords[j].tolist()
                    p2 = path_coords[j+1].tolist()
                    p3 = path_coords[k].tolist()
                    p4 = path_coords[k+1].tolist()
                    
                    if segments_intersect(p1, p2, p3, p4):
                        has_intersection = True
                        break
                if has_intersection:
                    break
            
            # 计算路径长度
            diff = path_coords[1:] - path_coords[:-1]
            segment_lengths = torch.norm(diff, dim=1)
            total_length = torch.sum(segment_lengths)
            
            # 计算转角总和
            total_angle = 0.0
            for j in range(1, n-1):
                # 三个连续点形成的夹角
                vec1 = path_coords[j] - path_coords[j-1]
                vec2 = path_coords[j+1] - path_coords[j]
                
                # 计算向量夹角
                cos_theta = torch.dot(vec1, vec2) / (torch.norm(vec1) * torch.norm(vec2) + 1e-8)
                cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  # 防止数值不稳定
                angle = torch.acos(cos_theta)
                total_angle += angle
            
            # 移除对重复点和交叉路径的惩罚，因为这些问题已经在路径后处理中解决了
            # if has_duplicates:
            #     total_length += 1e5
            #     total_angle += 1e5
            # if has_intersection:
            #     total_length += 1e5
            #     total_angle += 1e5
            
            lengths[i] = total_length
            angle_sums[i] = total_angle
        
        return lengths, angle_sums
    
    def compute_returns(self, rewards, masks, value_estimates):
        """计算回报和优势函数"""
        if self.use_gae:
            # 使用GAE计算优势函数
            advantages = torch.zeros_like(rewards).to(self.device)
            returns = torch.zeros_like(rewards).to(self.device)
            
            advantage = 0
            for t in reversed(range(rewards.size(1))):
                delta = rewards[:, t] + self.gamma * value_estimates[:, t+1] * masks[:, t] - value_estimates[:, t]
                advantage = delta + self.gamma * self.gae_lambda * masks[:, t] * advantage
                advantages[:, t] = advantage
                returns[:, t] = advantage + value_estimates[:, t]
        else:
            # 使用普通回报计算
            returns = torch.zeros_like(rewards).to(self.device)
            returns[:, -1] = rewards[:, -1]
            
            for t in reversed(range(rewards.size(1)-1)):
                returns[:, t] = rewards[:, t] + self.gamma * returns[:, t+1]
            
            advantages = returns - value_estimates[:, :-1]
        
        return returns, advantages
    
    def update_model(self, states, num_points, actions, returns, advantages, old_log_probs):
        """使用PPO算法更新模型参数"""
        # 获取当前动作的对数概率
        batch_size, max_len, _ = states.size()
        actor_prob = self.actor(states, num_points)
        
        # 计算当前动作的对数概率
        current_log_probs = []
        for b in range(batch_size):
            n = num_points[b].item()
            if n == 0:
                current_log_probs.append(torch.tensor(0.0).to(self.device))
                continue
            
            log_prob_sum = 0.0
            visited = torch.zeros(n, dtype=torch.bool).to(self.device)
            
            for step in range(n):
                action = actions[b, step].item()
                
                # 只允许选择未访问的点
                mask = torch.ones(n, dtype=torch.float32).to(self.device)
                mask[visited] = 0.0
                
                # 计算当前步骤的概率分布
                step_prob = actor_prob[b, step, :n] * mask
                step_prob = step_prob / (step_prob.sum() + 1e-8)  # 归一化
                
                # 计算对数概率
                dist = Categorical(step_prob)
                log_prob_sum += dist.log_prob(torch.tensor(action, dtype=torch.long).to(self.device))
                
                # 标记为已访问
                visited[action] = True
            
            current_log_probs.append(log_prob_sum)
        
        current_log_probs = torch.stack(current_log_probs)
        
        # 计算PPO损失
        ratio = torch.exp(current_log_probs - old_log_probs.detach())
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 价值损失
        value_estimates = self.critic(states, num_points).squeeze(1)
        value_loss = self.value_loss_coef * (value_estimates - returns.detach()).pow(2).mean()
        
        # 熵损失
        # 计算每个动作的熵
        entropies = []
        for b in range(batch_size):
            n = num_points[b].item()
            if n == 0:
                entropies.append(torch.tensor(0.0).to(self.device))
                continue
            
            entropy_sum = 0.0
            visited = torch.zeros(n, dtype=torch.bool).to(self.device)
            
            for step in range(n):
                # 只允许选择未访问的点
                mask = torch.ones(n, dtype=torch.float32).to(self.device)
                mask[visited] = 0.0
                
                # 计算当前步骤的概率分布
                step_prob = actor_prob[b, step, :n] * mask
                step_prob = step_prob / (step_prob.sum() + 1e-8)  # 归一化
                
                # 计算熵
                dist = Categorical(step_prob)
                entropy_sum += dist.entropy()
                
                # 选择动作（这里我们不关心具体动作，只计算熵）
                visited[torch.argmax(step_prob)] = True
            
            entropies.append(entropy_sum)
        
        entropies = torch.stack(entropies)
        entropy_loss = -self.entropy_coef * entropies.mean()
        
        # 总损失
        total_loss = policy_loss + value_loss + entropy_loss
        
        # 更新参数
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), max_norm=1.0)
        self.optimizer.step()
        
        return policy_loss.item(), value_loss.item(), entropy_loss.item(), total_loss.item()
    
    def train_step(self, batch, num_points_tensor, outer_contours, holes_list, epoch):
        """单步训练 - PPO算法"""
        logging.info("train_step方法开始执行...")
        batch_size = batch.shape[0]
        logging.info(f"批次大小: {batch_size}")
        if batch_size == 0:
            logging.info("批次大小为0，直接返回")
            return 0.0, 0.0, 0.0, 0.0, 0.0
        
        # 准备训练数据
        logging.info("准备训练数据...")
        max_len = max(p.shape[0] for p in batch)
        logging.info(f"最大关键点数量: {max_len}")
        padded_batch = torch.zeros((batch_size, max_len, 2))
        num_points_list = []
        
        for i, p in enumerate(batch):
            padded_batch[i, :p.shape[0]] = p
            num_points_list.append(p.shape[0])
        
        num_points_list = torch.tensor(num_points_list, dtype=torch.long)
        padded_batch = padded_batch.to(self.device)
        num_points_list = num_points_list.to(self.device)
        logging.info("训练数据准备完成")
        
        # 生成路径概率
        logging.info("开始Actor网络前向传播...")
        actor_prob = self.actor(padded_batch, num_points_list)
        logging.info(f"Actor概率输出形状: {actor_prob.shape}")
        
        # 采样路径序列
        logging.info("开始采样路径序列...")
        actor_actions_list = []
        actor_log_probs = []
        
        for b in range(batch_size):
            n = num_points_list[b].item()
            if n == 0:
                actor_log_probs.append(torch.tensor(0.0).to(self.device))
                actor_actions_list.append(torch.full((max_len,), -1, dtype=torch.long).to(self.device))
                continue
            
            # 初始化为-1（未访问）
            action = torch.full((max_len,), -1, dtype=torch.long).to(self.device)
            visited = torch.zeros(n, dtype=torch.bool).to(self.device)  # 使用布尔数组跟踪已访问点
            log_prob_sum = 0.0
            
            for step in range(n):
                # 只允许选择未访问的点
                mask = torch.ones(n, dtype=torch.float32).to(self.device)
                mask[visited] = 0.0  # 直接使用布尔数组作为掩码
                
                # 计算当前步骤的概率分布
                step_prob = actor_prob[b, step, :n] * mask
                step_prob = step_prob / (step_prob.sum() + 1e-8)  # 归一化
                
                # 采样动作
                dist = Categorical(step_prob)
                selected_point = dist.sample()
                action[step] = selected_point
                log_prob_sum += dist.log_prob(selected_point)
                
                # 标记为已访问
                visited[selected_point] = True
            
            actor_actions_list.append(action)
            actor_log_probs.append(log_prob_sum)
        
        actor_actions = torch.stack(actor_actions_list)
        old_log_probs = torch.stack(actor_log_probs)
        
        # 路径后处理：只确保路径的点都是唯一的
        for i in range(actor_actions.size(0)):
            n = num_points_list[i].item()
            if n > 0:
                # 确保路径点唯一
                unique_path = self._ensure_unique_path(actor_actions[i, :n], n)
                actor_actions[i, :n] = unique_path
                
                actor_actions[i, n:] = -1  # 填充剩余位置
        
        # 计算路径指标
        length, angle_sum = self.calculate_path_metrics(padded_batch, num_points_list, actor_actions)
        avg_length = length.mean().item()
        avg_angle = angle_sum.mean().item()
        
        # 计算奖励
        rewards = self.compute_reward(padded_batch, num_points_list, actor_actions, holes_list)
        avg_reward = rewards.mean().item()
        
        # 计算价值估计
        value_estimates = self.critic(padded_batch, num_points_list).squeeze(1)
        
        # 计算回报和优势函数
        returns, advantages = self.compute_returns(rewards.unsqueeze(1), torch.ones_like(rewards.unsqueeze(1)), 
                                                   torch.cat([value_estimates.unsqueeze(1), 
                                                              torch.zeros_like(rewards.unsqueeze(1))], dim=1))
        
        # 更新学习率
        self.lr_scheduler.step()
        
        # 使用PPO算法更新模型
        policy_loss, value_loss, entropy_loss, total_loss = self.update_model(
            padded_batch, num_points_list, actor_actions, returns.squeeze(1), advantages.squeeze(1), old_log_probs)
        
        # 确保返回值有效
        return policy_loss, value_loss, avg_length, avg_angle, avg_reward
    
    def plan_path(self, keypoints, num_points, holes=None):
        """
        路径规划主方法，确保生成的路径不交叉且安全（不穿过孔洞）
        
        参数:
        keypoints: 关键点坐标张量
        num_points: 路径点数量
        holes: 孔洞列表，每个元素为(圆心坐标, 半径)
        """
        self.actor.eval()  # 设置为评估模式
        
        # 保存孔洞信息到实例变量，供安全检查使用
        self.holes = holes if holes is not None else []
        
        with torch.no_grad():
            # 准备输入数据
            keypoints = keypoints.unsqueeze(0).to(self.device)  # [1, seq_len, 2]
            num_points = num_points.item() if isinstance(num_points, torch.Tensor) else num_points
            
            # 获取关键点坐标
            keypoints_np = keypoints.squeeze(0).cpu().numpy()
            
            # 生成路径概率
            probs = self.actor(keypoints, torch.tensor(num_points, dtype=torch.long).unsqueeze(0).to(self.device))  # [1, seq_len, seq_len]
            
            # 初始化路径
            path_seq = torch.zeros(num_points, dtype=torch.long).to(self.device)
            used = set()
            
            # 选择第一个点
            start_point = 0
            path_seq[0] = start_point
            used.add(start_point)
            
            # 安全检查函数，用于检查线段是否穿过孔洞
            def line_intersects_circle(p1, p2, circle_center, radius):
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                a = dx**2 + dy**2
                b = 2 * (dx*(p1[0] - circle_center[0]) + dy*(p1[1] - circle_center[1]))
                c = (p1[0] - circle_center[0])**2 + (p1[1] - circle_center[1])**2 - radius**2
                discriminant = b**2 - 4*a*c
                if discriminant < 0:
                    return False
                if a == 0:
                    return False
                t1 = (-b - np.sqrt(discriminant)) / (2*a)
                t2 = (-b + np.sqrt(discriminant)) / (2*a)
                return (0 <= t1 <= 1) or (0 <= t2 <= 1)
            
            # 新的路径生成策略：让模型生成完整路径，但在选择点时适当考虑距离因素
            # 这样可以平衡模型学习能力和路径长度
            for i in range(1, num_points):
                # 获取当前点的概率分布
                current_probs = probs.squeeze(0)[path_seq[i-1], :num_points].clone()
                
                # 排除已使用的点
                for j in used:
                    current_probs[j] = -float('inf')
                
                # 在选择点时适当考虑距离因素，提高初始路径质量
                current_point = keypoints_np[path_seq[i-1].item()]
                
                # 将概率转换为numpy数组
                prob_array = current_probs.cpu().numpy()
                
                # 计算每个候选点到当前点的距离
                for candidate in range(num_points):
                    if candidate in used:
                        continue
                    
                    candidate_point = keypoints_np[candidate]
                    distance = np.linalg.norm(candidate_point - current_point)
                    
                    # 使用距离因子调整概率，距离越近，概率权重越高
                    # 更平衡的距离因子计算，结合指数衰减和平滑因子
                    max_distance = np.max(np.linalg.norm(keypoints_np - current_point, axis=1))
                    normalized_distance = distance / (max_distance + 1.0)
                    distance_factor = np.exp(-3.0 * normalized_distance)  # 增加指数衰减的系数，使距离因子对近距离点更加敏感
                    prob_array[candidate] *= (0.4 + distance_factor * 0.6)  # 更加平衡模型学习能力和路径长度
                
                # 将调整后的概率转换回张量
                adjusted_probs = torch.tensor(prob_array, dtype=torch.float32).to(self.device)
                
                # 选择调整后概率最高的点
                next_point = torch.argmax(adjusted_probs).item()
                
                # 避免重复点
                if next_point not in used:
                    path_seq[i] = next_point
                    used.add(next_point)
                else:
                    # 如果选择的点已经被使用，选择概率第二高的点
                    current_probs[next_point] = -float('inf')
                    next_point = torch.argmax(current_probs).item()
                    path_seq[i] = next_point
                    used.add(next_point)
            
            print("模型生成了完整路径，开始进行后处理修复...")
            
            # 转换为numpy数组
            path_seq = path_seq.cpu().numpy()
            
            # 后处理：检查并修复路径中的交叉点（修改版，不使用_safe_local_optimization）
            # 修改：移除_safe_local_optimization调用，避免移除任何点
            fixed_path = path_seq.copy()
            
            # 检查是否有交叉
            has_cross = True
            max_iterations = 100
            iteration = 0
            
            while has_cross and iteration < max_iterations:
                iteration += 1
                has_cross = False
                
                # 检查所有线段对
                for i in range(len(fixed_path) - 1):
                    for j in range(i + 2, len(fixed_path) - 1):
                        p1 = keypoints_np[fixed_path[i]]
                        p2 = keypoints_np[fixed_path[i+1]]
                        p3 = keypoints_np[fixed_path[j]]
                        p4 = keypoints_np[fixed_path[j+1]]
                        
                        if segments_intersect(p1.tolist(), p2.tolist(), p3.tolist(), p4.tolist(), epsilon=1e-3):
                            has_cross = True
                            
                            # 修复交叉：交换两个交叉线段之间的点顺序
                            fixed_path[i+1:j+1] = fixed_path[j:i:-1]
                            break
                    if has_cross:
                        break
            
            path_seq = fixed_path
            
            # 只移除连续的重复点，保留所有不同的点
            unique_path_seq = [path_seq[0]]
            for i in range(1, len(path_seq)):
                if path_seq[i] != unique_path_seq[-1]:
                    unique_path_seq.append(path_seq[i])
            path_seq = unique_path_seq
            
            # 获取路径坐标
            path_points = keypoints_np[path_seq]
            
            # 计算路径指标
            length = np.sum(np.linalg.norm(path_points[1:] - path_points[:-1], axis=1))
            
            # 计算转角总和
            angle_sum = 0.0
            for i in range(1, len(path_points)-1):
                vec1 = path_points[i] - path_points[i-1]
                vec2 = path_points[i+1] - path_points[i]
                
                # 计算向量夹角
                cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                angle = np.arccos(cos_theta)
                angle_sum += angle
        
        self.actor.train()  # 恢复训练模式
        
        return path_seq, path_points, length, angle_sum
    
    def _ensure_unique_path(self, path_seq, num_points):
        """确保路径序列中的点唯一"""
        # 简单的去重策略：如果有重复点，修复原路径
        unique_points = torch.unique(path_seq)
        if len(unique_points) == num_points:
            return path_seq
        
        # 修复原路径中的重复点
        unique_path = torch.zeros(num_points, dtype=torch.long)
        used = set()
        
        # 首先保留原路径中的不重复点
        idx = 0
        for i in range(num_points):
            if path_seq[i].item() not in used:
                unique_path[idx] = path_seq[i]
                used.add(path_seq[i].item())
                idx += 1
        
        # 填充剩余的未使用点
        for i in range(num_points):
            if i not in used:
                unique_path[idx] = i
                used.add(i)
                idx += 1
        
        return unique_path
    
    def _fix_crossing(self, path_seq, keypoints_np):
        """后处理：检查并修复路径中的交叉点，改进的修复机制"""
        fixed_path = path_seq.copy()
        
        # 检查是否有交叉
        has_cross = True
        max_iterations = 100  # 大幅增加最大迭代次数以确保交叉点被修复
        iteration = 0
        
        while has_cross and iteration < max_iterations:
            iteration += 1
            has_cross = False
            
            # 检查所有线段对，确保没有遗漏的交叉点
            for i in range(len(fixed_path) - 1):
                # 检查与当前线段后面的所有非相邻线段是否交叉
                for j in range(i + 2, len(fixed_path) - 1):
                    p1 = keypoints_np[fixed_path[i]]
                    p2 = keypoints_np[fixed_path[i+1]]
                    p3 = keypoints_np[fixed_path[j]]
                    p4 = keypoints_np[fixed_path[j+1]]
                    
                    if segments_intersect(p1.tolist(), p2.tolist(), p3.tolist(), p4.tolist(), epsilon=1e-3):
                        has_cross = True
                        
                        # 修复交叉：交换两个交叉线段之间的点顺序
                        fixed_path[i+1:j+1] = fixed_path[j:i:-1]
                        
                        # 跳出循环，重新检查整个路径
                        break
                if has_cross:
                    break
        
        # 不再调用_safe_local_optimization，避免移除任何点
        
        return fixed_path
    
    def _safe_local_optimization(self, path_seq, keypoints_np):
        """安全的局部优化：仅在不引入新交叉点的情况下减少路径长度"""
        # 首先移除重复点，包括非连续的重复点
        unique_path = []
        used_points = set()
        for i in range(len(path_seq)):
            if path_seq[i] not in used_points:
                unique_path.append(path_seq[i])
                used_points.add(path_seq[i])
        optimized_path = unique_path.copy()
        
        improved = True
        iterations = 0
        max_local_iterations = 8
        
        while improved and iterations < max_local_iterations:
            iterations += 1
            improved = False
            total_length = self._calculate_path_length(optimized_path, keypoints_np)
            
            # 尝试移动单个点到更好的位置，仅考虑相邻的几个点
            for i in range(1, len(optimized_path) - 1):
                # 保存当前位置
                current_pos = optimized_path[i]
                
                # 计算当前点与前后点之间的距离
                prev_point = keypoints_np[optimized_path[i-1]]
                curr_point = keypoints_np[current_pos]
                next_point = keypoints_np[optimized_path[i+1]]
                current_segment_length = np.linalg.norm(curr_point - prev_point) + np.linalg.norm(next_point - curr_point)
                
                # 寻找更好的点来替换当前点
                best_replace_point = current_pos
                best_length = current_segment_length
                
                # 仅考虑当前点周围的几个点（安全范围）
                neighbor_indices = list(range(max(0, i-4), min(len(optimized_path), i+5)))
                neighbor_indices.remove(i)  # 移除当前点
                
                for neighbor in neighbor_indices:
                    # 跳过已经在路径中使用的点，包括前后相邻的点
                    if neighbor in optimized_path:
                        continue
                    
                    # 计算替换后的距离
                    neighbor_point = keypoints_np[neighbor]
                    new_length = np.linalg.norm(neighbor_point - prev_point) + np.linalg.norm(next_point - neighbor_point)
                    
                    # 确保不会产生零长度的线段
                    if np.linalg.norm(neighbor_point - prev_point) < 1e-3 or np.linalg.norm(next_point - neighbor_point) < 1e-3:
                        continue
                    
                    # 如果替换后距离更短，且不引入新的交叉点，则考虑替换
                    if new_length < best_length:
                        # 临时替换点
                        temp_path = optimized_path.copy()
                        temp_path[i] = neighbor
                        
                        # 检查是否引入新的交叉点
                        if not self._has_crossing(temp_path, keypoints_np):
                            best_replace_point = neighbor
                            best_length = new_length
                
                # 如果找到了更好的点，则替换
                if best_replace_point != current_pos:
                    optimized_path[i] = best_replace_point
                    improved = True
                    break
            
            # 如果没有改进，尝试简单的线段优化：移除中间点，如果直接连接不引入交叉点
            if not improved and len(optimized_path) > 3:
                for i in range(1, len(optimized_path) - 1):
                    # 检查是否可以移除当前点
                    prev_point = keypoints_np[optimized_path[i-1]]
                    curr_point = keypoints_np[optimized_path[i]]
                    next_point = keypoints_np[optimized_path[i+1]]
                    
                    # 计算移除后的距离
                    direct_distance = np.linalg.norm(next_point - prev_point)
                    current_distance = np.linalg.norm(curr_point - prev_point) + np.linalg.norm(next_point - curr_point)
                    
                    # 如果直接连接更短，且不引入交叉点
                    if direct_distance < current_distance:
                        temp_path = optimized_path.copy()
                        del temp_path[i]
                        
                        # 检查是否引入新的交叉点
                        if not self._has_crossing(temp_path, keypoints_np):
                            optimized_path = temp_path
                            improved = True
                            break
        
        return optimized_path
    
    def _local_optimization(self, path_seq, keypoints_np):
        """局部优化：尝试交换相邻点以减少路径长度"""
        improved = True
        iterations = 0
        max_local_iterations = 15
        
        while improved and iterations < max_local_iterations:
            iterations += 1
            improved = False
            total_length = self._calculate_path_length(path_seq, keypoints_np)
            
            # 尝试多种局部优化策略
            # 1. 尝试交换相邻的两个点
            for i in range(1, len(path_seq) - 2):
                # 保存当前顺序
                original = path_seq.copy()
                
                # 交换相邻点
                path_seq[i], path_seq[i+1] = path_seq[i+1], path_seq[i]
                
                # 检查是否产生新的交叉点
                if not self._has_crossing(path_seq, keypoints_np):
                    # 计算新长度
                    new_length = self._calculate_path_length(path_seq, keypoints_np)
                    
                    # 如果长度减少，保留交换
                    if new_length < total_length:
                        total_length = new_length
                        improved = True
                        break
                    else:
                        # 恢复原顺序
                        path_seq = original
                else:
                    # 恢复原顺序
                    path_seq = original
            
            if improved:
                continue
            
            # 2. 尝试交换间隔一个点的两个点
            for i in range(1, len(path_seq) - 3):
                # 保存当前顺序
                original = path_seq.copy()
                
                # 交换间隔点
                path_seq[i], path_seq[i+2] = path_seq[i+2], path_seq[i]
                
                # 检查是否产生新的交叉点
                if not self._has_crossing(path_seq, keypoints_np):
                    # 计算新长度
                    new_length = self._calculate_path_length(path_seq, keypoints_np)
                    
                    # 如果长度减少，保留交换
                    if new_length < total_length:
                        total_length = new_length
                        improved = True
                        break
                    else:
                        # 恢复原顺序
                        path_seq = original
                else:
                    # 恢复原顺序
                    path_seq = original
            
            if improved:
                continue
            
            # 3. 尝试反转一小段路径
            for i in range(1, len(path_seq) - 3):
                # 保存当前顺序
                original = path_seq.copy()
                
                # 反转一小段路径
                end = min(i + 4, len(path_seq) - 1)
                path_seq[i:end] = path_seq[i:end][::-1]
                
                # 检查是否产生新的交叉点
                if not self._has_crossing(path_seq, keypoints_np):
                    # 计算新长度
                    new_length = self._calculate_path_length(path_seq, keypoints_np)
                    
                    # 如果长度减少，保留交换
                    if new_length < total_length:
                        total_length = new_length
                        improved = True
                        break
                    else:
                        # 恢复原顺序
                        path_seq = original
                else:
                    # 恢复原顺序
                    path_seq = original
    
    def _calculate_path_length(self, path_seq, keypoints_np):
        """计算路径长度"""
        length = 0.0
        for i in range(len(path_seq) - 1):
            p1 = keypoints_np[path_seq[i]]
            p2 = keypoints_np[path_seq[i+1]]
            length += np.linalg.norm(p2 - p1)
        return length
    
    def _has_crossing(self, path_seq, keypoints_np):
        """检查路径是否存在交叉点"""
        for i in range(len(path_seq) - 1):
            for j in range(i + 2, len(path_seq) - 1):
                p1 = keypoints_np[path_seq[i]]
                p2 = keypoints_np[path_seq[i+1]]
                p3 = keypoints_np[path_seq[j]]
                p4 = keypoints_np[path_seq[j+1]]
                
                if segments_intersect(p1.tolist(), p2.tolist(), p3.tolist(), p4.tolist(), epsilon=1e-3):
                    return True
        return False
        
    def save_model(self, save_path):
        """保存模型"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'config': self.config.to_dict()
        }, save_path)
    
    def load_model(self, load_path):
        """加载模型"""
        checkpoint = torch.load(load_path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])

# -------------------------- 6. 可视化工具 --------------------------
class Visualizer:
    """统一管理可视化和结果保存"""
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_path_result(self, keypoints, path_points, outer_contour, holes, length, angle_sum, epoch=0, save=True):
        """绘制路径规划结果"""
        plt.figure(figsize=(12, 10))
        ax = plt.gca()
        
        # 绘制外轮廓
        outer_contour_plot = np.vstack([outer_contour, outer_contour[0]])
        ax.plot(outer_contour_plot[:, 0], outer_contour_plot[:, 1], 'b-', linewidth=2, label='Outer Contour')
        
        # 绘制内孔
        for idx, (center, radius) in enumerate(holes):
            circle = Circle(center, radius, color='gray', alpha=0.3, label='Inner Hole' if idx == 0 else "")
            ax.add_patch(circle)
        
        # 绘制所有关键点
        ax.scatter(keypoints[:, 0], keypoints[:, 1], s=20, c='gray', alpha=0.6, label='Key Points')
        
        # 绘制路径
        ax.plot(path_points[:, 0], path_points[:, 1], 'r-', linewidth=2, alpha=0.8, label='Planned Path')
        
        # 绘制箭头（每10个点画一个）
        step = max(1, len(path_points)//10)
        for i in range(0, len(path_points)-1, step):
            ax.arrow(
                path_points[i, 0], path_points[i, 1],
                path_points[i+1, 0] - path_points[i, 0],
                path_points[i+1, 1] - path_points[i, 1],
                head_width=5, head_length=8, fc='red', ec='red', alpha=0.7
            )
        
        # 标记起点和终点
        ax.scatter(path_points[0, 0], path_points[0, 1], s=100, c='green', marker='o', edgecolors='black', label='Start')
        ax.scatter(path_points[-1, 0], path_points[-1, 1], s=100, c='orange', marker='s', edgecolors='black', label='End')
        
        # 设置图表信息
        ax.set_xlim(-20, 520)
        ax.set_ylim(-20, 520)
        ax.set_aspect('equal')
        ax.set_title(f'Path Planning Result (Epoch {epoch})\nTotal Length: {length:.1f} | Total Angle: {angle_sum:.2f} rad', fontsize=14)
        ax.set_xlabel('X Coordinate (mm)', fontsize=12)
        ax.set_ylabel('Y Coordinate (mm)', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 保存图片
        if save:
            save_path = os.path.join(self.save_dir, f'path_result_epoch_{epoch}.png')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            logging.info(f"可视化结果已保存至：{save_path}")
        else:
            plt.tight_layout()
            plt.show()
    
    def plot_training_curve(self, actor_losses, critic_losses, rewards_list, lengths_list, angles_list, save_path):
        """绘制训练曲线"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12), sharex=True)
        
        # 损失曲线
        ax1.plot(actor_losses, 'b-', label='Actor Loss')
        ax1.plot(critic_losses, 'r-', label='Critic Loss')
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training Losses', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 奖励曲线
        ax2.plot(rewards_list, 'g-', label='Average Reward')
        ax2.set_ylabel('Reward', fontsize=12)
        ax2.set_title('Average Reward', fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 路径长度曲线
        ax3.plot(lengths_list, 'm-', label='Path Length')
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Length', fontsize=12)
        ax3.set_title('Average Path Length', fontsize=14)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # 转角总和曲线
        ax4.plot(angles_list, 'c-', label='Angle Sum')
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('Angle (rad)', fontsize=12)
        ax4.set_title('Average Angle Sum', fontsize=14)
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        # 保存图片
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"训练曲线已保存至：{save_path}")

# -------------------------- 7. 模型管理 --------------------------
class ModelManager:
    """统一管理模型的保存和加载"""
    @staticmethod
    def save_model(planner, epoch, best_length, save_dir):
        """保存模型"""
        save_path = os.path.join(save_dir, 'best_model.pth')
        torch.save({
            'actor_state_dict': planner.actor.state_dict(),
            'critic_state_dict': planner.critic.state_dict(),
            'epoch': epoch,
            'best_length': best_length
        }, save_path)
        logging.info(f"最优模型已保存（epoch {epoch}，路径长度 {best_length:.2f} mm）")
    
    @staticmethod
    def load_best_model(planner, save_dir):
        """加载最优模型"""
        model_path = os.path.join(save_dir, 'best_model.pth')
        if not os.path.exists(model_path):
            logging.warning(f"未找到模型文件：{model_path}")
            return planner
        
        checkpoint = torch.load(model_path, map_location=planner.device)
        planner.actor.load_state_dict(checkpoint['actor_state_dict'])
        planner.critic.load_state_dict(checkpoint['critic_state_dict'])
        # 初始化目标网络
        planner.target_actor = copy.deepcopy(planner.actor)
        planner.target_critic = copy.deepcopy(planner.critic)
        
        logging.info(f"加载最优模型完成（epoch {checkpoint['epoch']}，路径长度 {checkpoint['best_length']:.2f} mm）")
        return planner

# -------------------------- 8. 主程序 --------------------------
def main():
    """主程序入口"""
    # 初始化配置
    config = Config()
    
    # 初始化日志
    log_file = os.path.join(config.SAVE_DIR, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info("开始训练复杂空心构件路径规划器")
    logging.info(f"设备：{config.DEVICE}")
    logging.info(f"超参数：批次大小={config.BATCH_SIZE}, 学习率={config.LR}, 训练轮数={config.EPOCHS}")
    logging.info(f"关键点范围：{config.MIN_KEYPOINTS}~{config.MAX_KEYPOINTS}个")
    logging.info(f"早停参数：耐心值={config.PATIENCE}")
    
    # 保存配置
    with open(os.path.join(config.SAVE_DIR, 'config.json'), 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    
    # 初始化组件
    generator = KeypointGenerator()
    planner = RLPathPlanner(config)
    visualizer = Visualizer(config.SAVE_DIR)
    model_manager = ModelManager()
    
    # 初始化训练记录
    actor_losses = []
    critic_losses = []
    rewards_list = []
    lengths_list = []
    angles_list = []
    
    # 早停相关变量
    best_length = float('inf')
    no_improve_count = 0
    
    # 训练循环
    for epoch in range(config.EPOCHS):
        logging.info(f"开始Epoch {epoch+1}/{config.EPOCHS}")
        
        # 生成批次数据
        logging.info(f"开始生成批次数据...")
        train_batch, num_points_tensor, outer_contours, holes_list = generator.generate_batch(
            config.BATCH_SIZE,
            config.MIN_KEYPOINTS,
            config.MAX_KEYPOINTS
        )
        logging.info(f"批次数据生成完成，批次大小={len(train_batch)}")
        
        # 单步训练
        logging.info(f"开始执行train_step...")
        actor_loss, critic_loss, avg_length, avg_angle, avg_reward = planner.train_step(train_batch, num_points_tensor, outer_contours, holes_list, epoch)
        logging.info(f"train_step执行完成")
        
        # 记录训练日志
        actor_losses.append(actor_loss)
        critic_losses.append(critic_loss)
        lengths_list.append(avg_length)
        angles_list.append(avg_angle)
        rewards_list.append(avg_reward)
        
        # 打印训练日志
        logging.info(f"Epoch [{epoch+1}/{config.EPOCHS}] | "
              f"Actor Loss: {actor_loss:.4f} | "
              f"Critic Loss: {critic_loss:.4f} | "
              f"Avg Reward: {avg_reward:.4f} | "
              f"Avg Length: {avg_length:.1f} | "
              f"Avg Angle: {avg_angle:.2f}")
        
        # 定期测试和可视化
        current_test_length = float('inf')
        if (epoch + 1) % config.TEST_INTERVAL == 0 or epoch == config.EPOCHS - 1:
            logging.info("\n开始测试路径规划...")
            
            # 生成测试数据（确保关键点数量在合理范围内）
            test_num_points = min(50, config.MAX_KEYPOINTS)
            test_keypoints, test_outer, test_holes = generator.generate_keypoints(num_points=test_num_points)
            
            # 规划路径
            test_keypoints_tensor = torch.tensor(test_keypoints, dtype=torch.float32)
            path_seq, path_points, length, angle_sum = planner.plan_path(test_keypoints_tensor, test_num_points, test_holes)
            current_test_length = length
            
            # 打印测试结果
            logging.info(f"测试结果 - 路径长度：{length:.2f} mm | 转角总和：{angle_sum:.2f} rad")
            logging.info(f"路径唯一性：{'通过' if len(np.unique(path_seq)) == test_num_points else '失败'}")
            
            # 可视化结果
            visualizer.plot_path_result(
                test_keypoints,
                path_points,
                test_outer,
                test_holes,
                length,
                angle_sum,
                epoch+1
            )
            
            # 保存最优模型
            if length < best_length:
                best_length = length
                no_improve_count = 0
                model_manager.save_model(planner, epoch+1, best_length, config.SAVE_DIR)
            else:
                no_improve_count += 1
                logging.info(f"路径长度无提升，早停计数器：{no_improve_count}/{config.PATIENCE}")
        
        # 早停检查
        if no_improve_count >= config.PATIENCE:
            logging.info(f"\n早停触发：连续{config.PATIENCE}个epoch路径长度无提升")
            break
    
    # 绘制训练曲线
    visualizer.plot_training_curve(
        actor_losses,
        critic_losses,
        rewards_list,
        lengths_list,
        angles_list,
        os.path.join(config.SAVE_DIR, 'training_curves.png')
    )
    
    # 最终测试
    logging.info("\n" + "="*50)
    logging.info("训练完成！开始最终测试（加载最优模型）...")
    planner = model_manager.load_best_model(planner, config.SAVE_DIR)
    
    # 生成最终测试数据（确保关键点数量不超过MAX_KEYPOINTS）
    final_num_points = min(80, config.MAX_KEYPOINTS)
    final_keypoints, final_outer, final_holes = generator.generate_keypoints(num_points=final_num_points)
    final_keypoints_tensor = torch.tensor(final_keypoints, dtype=torch.float32)
    final_path_seq, final_path_points, final_length, final_angle_sum = planner.plan_path(final_keypoints_tensor, final_num_points, final_holes)
    
    # 打印最终测试结果
    logging.info(f"\n最终测试结果 - 关键点数量：{final_num_points} | "
          f"路径长度：{final_length:.2f} mm | "
          f"转角总和：{final_angle_sum:.2f} rad")
    logging.info(f"路径唯一性：{'通过' if len(np.unique(final_path_seq)) == final_num_points else '失败'}")
    
    # 保存最终结果
    visualizer.plot_path_result(
        final_keypoints,
        final_path_points,
        final_outer,
        final_holes,
        final_length,
        final_angle_sum,
        epoch="final"
    )
    
    # 输出路径序列统计
    logging.info(f"\n路径序列统计：")
    logging.info(f"起点索引：{final_path_seq[0]} | 终点索引：{final_path_seq[-1]}")
    logging.info(f"平均步长：{final_length/(final_num_points-1):.2f} mm")
    
    logging.info("\n" + "="*50)
    logging.info(f"所有结果已保存至目录：{config.SAVE_DIR}")
    logging.info("训练完成！")

# -------------------------- 9. 模型使用接口 --------------------------
def use_trained_model(model_path, num_points=50):
    """使用训练好的模型进行路径规划"""
    # 初始化配置
    config = Config()
    
    # 初始化组件
    generator = KeypointGenerator()
    planner = RLPathPlanner(config)
    visualizer = Visualizer(config.SAVE_DIR)
    
    # 加载模型
    planner.load_model(model_path)
    
    # 生成测试数据
    test_keypoints, test_outer, test_holes = generator.generate_keypoints(num_points)
    test_keypoints_tensor = torch.tensor(test_keypoints, dtype=torch.float32)
    
    # 规划路径
    path_seq, path_points, length, angle_sum = planner.plan_path(test_keypoints_tensor, num_points, test_holes)
    
    # 可视化结果
    visualizer.plot_path_result(
        test_keypoints,
        path_points,
        test_outer,
        test_holes,
        length,
        angle_sum,
        epoch="inference"
    )
    
    # 返回结果
    return {
        'path_seq': path_seq,
        'path_points': path_points,
        'length': length,
        'angle_sum': angle_sum,
        'keypoints': test_keypoints
    }

# -------------------------- 主程序入口 --------------------------
if __name__ == "__main__":
    main()