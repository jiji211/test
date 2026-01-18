import math
import numpy as np
import matplotlib.pyplot as plt
import os
import platform
import random
from typing import List, Tuple, Dict, Optional, Set

# 定义数据结构：G代码点（包含坐标和原始指令）
class GCodePoint:
    def __init__(self, x: float, y: float, z: float = 0.0, cmd: str = ""):
        self.x = round(x, 4)
        self.y = round(y, 4)
        self.z = round(z, 4)
        self.cmd = cmd  # 原始G代码指令
        self.visited = False  # 标记是否已加入路径

    def __repr__(self):
        return f"GCodePoint(x={self.x}, y={self.y}, z={self.z}, cmd='{self.cmd}')"

    def distance_to(self, other: "GCodePoint") -> float:
        """计算当前点到另一点的欧氏距离（支持2D/3D）"""
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return math.sqrt(dx**2 + dy**2 + dz**2)  # 通用欧氏距离公式

    @staticmethod
    def calculate_curvature_angle(p1: "GCodePoint", p2: "GCodePoint", p3: "GCodePoint") -> float:
        """计算三点形成的曲率夹角（p2为顶点），返回角度值"""
        vec1 = (p1.x - p2.x, p1.y - p2.y)
        vec2 = (p3.x - p2.x, p3.y - p2.y)
        
        # 计算向量点积和模长
        dot = vec1[0] * vec2[0] + vec1[1] * vec2[1]
        mag1 = math.hypot(vec1[0], vec1[1])  # 2D向量模长（hypot支持2参数）
        mag2 = math.hypot(vec2[0], vec2[1])
        
        if mag1 == 0 or mag2 == 0:
            return 180.0  # 共点时夹角为180°
        
        # 计算夹角（弧度转角度），限制值域避免acos报错
        cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
        angle_rad = math.acos(cos_angle)
        angle_deg = math.degrees(angle_rad)
        return angle_deg

class MultiHeadPathPlanner:
    def __init__(self, theta_th: float = 50.0, lambda1: float = 1.9, lambda2: float = 0.3):
        """
        初始化多喷头路径规划器
        :param theta_th: 曲率惩罚阈值（角度），默认50°
        :param lambda1: 曲率惩罚权重系数
        :param lambda2: 启停惩罚权重系数
        """
        self.theta_th = theta_th
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.path = []  # 最终规划路径
        self.raw_points = []  # 原始离散点
        
        # NSGA-II算法参数
        self.pop_size = 10  # 种群大小（进一步减小以加快计算）
        self.max_gen = 5  # 最大迭代次数（进一步减小以加快计算）
        self.crossover_prob = 0.9  # 交叉概率
        self.mutation_prob = 0.1  # 变异概率
        self.tournament_size = 2  # 锦标赛选择大小

    def get_desktop_path(self) -> str:
        """获取系统桌面路径（兼容Windows/macOS/Linux）"""
        if platform.system() == "Windows":
            # Windows桌面路径
            desktop = os.path.join(os.environ["USERPROFILE"], "Desktop")
        elif platform.system() == "Darwin":  # macOS
            desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        else:  # Linux
            desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        return desktop

    def read_gcode_file(self, filename: str = "left_region_gcode.gcode") -> List[str]:
        """读取桌面的G代码文件"""
        desktop_path = self.get_desktop_path()
        file_path = os.path.join(desktop_path, filename)
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"G代码文件未找到：{file_path}\n请确认文件是否在桌面，且文件名正确")
        
        # 读取文件（按行读取，忽略空行和注释）
        gcode_lines = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith(";"):  # 仅保留非空、非注释行
                    gcode_lines.append(line)
        
        print(f"成功读取G代码文件：{file_path}")
        print(f"有效G代码行数：{len(gcode_lines)}")
        return gcode_lines

    def parse_gcode(self, gcode_lines: List[str]) -> List[GCodePoint]:
        """解析G代码，提取离散坐标点"""
        points = []
        for line in gcode_lines:
            # 提取X/Y/Z坐标（G代码格式：G01 X10.0 Y20.0 Z0.0 F100）
            x = y = z = 0.0
            parts = line.split()
            for part in parts:
                part_upper = part.upper()  # 兼容小写G代码
                if part_upper.startswith("X"):
                    x = float(part_upper[1:])
                elif part_upper.startswith("Y"):
                    y = float(part_upper[1:])
                elif part_upper.startswith("Z"):
                    z = float(part_upper[1:])
            
            points.append(GCodePoint(x, y, z, line))
        
        self.raw_points = points  # 保存原始点
        print(f"解析出离散坐标点数量：{len(points)}")
        return points

    def calculate_objectives(self, individual: List[GCodePoint]) -> Tuple[float, float, float]:
        """
        计算个体的多目标值（NSGA-II使用）
        :param individual: 路径点列表（个体）
        :return: 三个目标值（需要最小化）
        """
        if len(individual) < 2:
            return (0.0, 0.0, 0.0)
        
        total_distance = 0.0  # 目标1：总路径长度（最小化）
        total_curvature_penalty = 0.0  # 目标2：总曲率惩罚（最小化）
        total_stop_penalty = 0.0  # 目标3：总启停惩罚（最小化）
        
        stop_threshold = 5.0  # 距离超过该阈值视为启停事件
        
        for i in range(len(individual) - 1):
            current_point = individual[i]
            next_point = individual[i+1]
            
            # 计算距离
            distance = current_point.distance_to(next_point)
            total_distance += distance
            
            # 计算曲率惩罚（仅当i>0时）
            if i > 0:
                prev_point = individual[i-1]
                angle = GCodePoint.calculate_curvature_angle(prev_point, current_point, next_point)
                if angle < self.theta_th:  # 锐角触发惩罚
                    total_curvature_penalty += self.lambda1 * (self.theta_th - angle) / self.theta_th
            
            # 计算启停惩罚
            if distance > stop_threshold:
                total_stop_penalty += self.lambda2 * (distance - stop_threshold) / stop_threshold
        
        # 检测交叉惩罚（目标4：交叉次数）
        total_crossing_penalty = 0
        for i in range(len(individual) - 1):
            for j in range(i + 2, len(individual) - 1):
                seg1_a, seg1_b = individual[i], individual[i+1]
                seg2_a, seg2_b = individual[j], individual[j+1]
                if self.segments_intersect(seg1_a, seg1_b, seg2_a, seg2_b):
                    total_crossing_penalty += 1
        
        # 返回需要最小化的目标值
        return (total_distance, total_curvature_penalty, total_stop_penalty, total_crossing_penalty)
    
    def calculate_fitness(self, current_point: GCodePoint, next_candidate: GCodePoint, prev_point: GCodePoint = None) -> float:
        """
        计算候选点的适应度值（用于贪心算法，保持兼容性）
        :param current_point: 当前路径最后一点
        :param next_candidate: 候选下一点
        :param prev_point: 当前路径倒数第二点（用于曲率计算）
        :return: 适应度值（越大越好）
        """
        # 1. 路径连续性项：点间距离的倒数（距离越小，值越大）
        distance = current_point.distance_to(next_candidate)
        continuity = 1.0 / (distance + 1e-6)  # 加小值避免除零
        
        # 2. 曲率惩罚项（仅当已有至少两个点时计算）
        curvature_penalty = 0.0
        if prev_point is not None:
            angle = GCodePoint.calculate_curvature_angle(prev_point, current_point, next_candidate)
            if angle < self.theta_th:  # 锐角触发惩罚
                curvature_penalty = self.lambda1 * (self.theta_th - angle) / self.theta_th
        
        # 3. 启停惩罚项：候选点与当前点是否为"启停断点"
        stop_penalty = 0.0
        stop_threshold = 5.0  # 距离超过该阈值视为启停事件
        if distance > stop_threshold:
            stop_penalty = self.lambda2 * (distance - stop_threshold) / stop_threshold
        
        # 总适应度 = 连续性 - 曲率惩罚 - 启停惩罚（惩罚项越小，适应度越高）
        fitness = continuity - curvature_penalty - stop_penalty
        return fitness

    @staticmethod
    def ccw(A: GCodePoint, B: GCodePoint, C: GCodePoint) -> bool:
        """判断三点是否为逆时针方向（线段交叉检测辅助函数）"""
        return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x)

    def segments_intersect(self, A: GCodePoint, B: GCodePoint, C: GCodePoint, D: GCodePoint) -> bool:
        """
        判断线段AB和线段CD是否相交（不包含端点重合的情况）
        :param A,B: 第一条线段的两个端点
        :param C,D: 第二条线段的两个端点
        :return: True=相交，False=不相交
        """
        # 排除端点重合的情况（避免误判相邻线段）
        if (A.x == C.x and A.y == C.y) or (A.x == D.x and A.y == D.y) or \
           (B.x == C.x and B.y == C.y) or (B.x == D.x and B.y == D.y):
            return False
        
        # 核心交叉检测算法
        return (MultiHeadPathPlanner.ccw(A, C, D) != MultiHeadPathPlanner.ccw(B, C, D)) and \
               (MultiHeadPathPlanner.ccw(A, B, C) != MultiHeadPathPlanner.ccw(A, B, D))

    def is_crossing_path(self, current_point: GCodePoint, candidate_point: GCodePoint) -> bool:
        """
        检查从当前点到候选点的线段是否与已规划路径的线段相交
        :param current_point: 当前路径最后一点
        :param candidate_point: 候选下一点
        :return: True=存在交叉，False=无交叉
        """
        # 路径长度不足2时无交叉可能
        if len(self.path) < 2:
            return False
        
        # 遍历已规划路径的所有线段，检查是否与候选线段相交
        candidate_segment = (current_point, candidate_point)
        for i in range(len(self.path) - 1):
            existing_segment = (self.path[i], self.path[i+1])
            if self.segments_intersect(
                existing_segment[0], existing_segment[1],
                candidate_segment[0], candidate_segment[1]
            ):
                return True
        return False
    
    def non_dominated_sorting(self, population: List[List[GCodePoint]]) -> List[List[int]]:
        """
        NSGA-II非支配排序
        :param population: 种群（个体列表）
        :return: 非支配等级列表，每个等级包含个体索引
        """
        # 计算每个个体的目标值
        objectives = [self.calculate_objectives(individual) for individual in population]
        n_individuals = len(population)
        
        # 初始化支配数和支配集
        domination_count = [0] * n_individuals  # 每个个体被多少个体支配
        dominated_set = [[] for _ in range(n_individuals)]  # 每个个体支配的个体集合
        
        # 计算支配关系
        for i in range(n_individuals):
            for j in range(n_individuals):
                if i == j:
                    continue
                    
                # 检查i是否支配j
                i_dominates_j = True
                j_dominates_i = True
                
                for obj_i, obj_j in zip(objectives[i], objectives[j]):
                    if obj_i > obj_j:  # 目标需要最小化
                        i_dominates_j = False
                    if obj_j > obj_i:
                        j_dominates_i = False
                
                if i_dominates_j and not j_dominates_i:
                    dominated_set[i].append(j)
                    domination_count[j] += 1
        
        # 初始化非支配等级
        frontiers = []  # 存储每个非支配等级的个体索引
        current_frontier = [i for i in range(n_individuals) if domination_count[i] == 0]
        
        while current_frontier:
            frontiers.append(current_frontier)
            next_frontier = []
            
            for i in current_frontier:
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_frontier.append(j)
            
            current_frontier = next_frontier
        
        return frontiers
    
    def crowding_distance_calculation(self, population: List[List[GCodePoint]], frontiers: List[List[int]]) -> List[float]:
        """
        NSGA-II拥挤度计算
        :param population: 种群（个体列表）
        :param frontiers: 非支配等级列表
        :return: 每个个体的拥挤度值
        """
        n_individuals = len(population)
        crowding_distance = [0.0] * n_individuals
        objectives = [self.calculate_objectives(individual) for individual in population]
        n_objectives = len(objectives[0]) if objectives else 0
        
        for frontier in frontiers:
            for obj_idx in range(n_objectives):
                # 按当前目标对前沿个体排序
                sorted_indices = sorted(frontier, key=lambda i: objectives[i][obj_idx])
                
                # 设置边界个体的拥挤度为无穷大
                crowding_distance[sorted_indices[0]] = float('inf')
                crowding_distance[sorted_indices[-1]] = float('inf')
                
                # 计算目标范围
                if n_objectives > 0:
                    obj_min = objectives[sorted_indices[0]][obj_idx]
                    obj_max = objectives[sorted_indices[-1]][obj_idx]
                    obj_range = obj_max - obj_min if obj_max != obj_min else 1.0
                else:
                    obj_range = 1.0
                
                # 计算中间个体的拥挤度
                for i in range(1, len(sorted_indices) - 1):
                    distance = objectives[sorted_indices[i+1]][obj_idx] - objectives[sorted_indices[i-1]][obj_idx]
                    crowding_distance[sorted_indices[i]] += distance / obj_range
        
        return crowding_distance
    
    def tournament_selection(self, population: List[List[GCodePoint]], frontiers: List[List[int]], crowding_distance: List[float]) -> List[GCodePoint]:
        """
        锦标赛选择
        :param population: 种群
        :param frontiers: 非支配等级列表
        :param crowding_distance: 拥挤度列表
        :return: 选中的个体
        """
        # 创建前沿到个体索引的映射
        individual_frontier = [0] * len(population)
        for frontier_idx, frontier in enumerate(frontiers):
            for individual_idx in frontier:
                individual_frontier[individual_idx] = frontier_idx
        
        # 随机选择锦标赛个体
        tournament = random.sample(range(len(population)), self.tournament_size)
        
        # 选择最优个体
        best_idx = tournament[0]
        for idx in tournament[1:]:
            # 比较前沿等级
            if individual_frontier[idx] < individual_frontier[best_idx]:
                best_idx = idx
            elif individual_frontier[idx] == individual_frontier[best_idx]:
                # 前沿等级相同，比较拥挤度
                if crowding_distance[idx] > crowding_distance[best_idx]:
                    best_idx = idx
        
        return population[best_idx]
    
    def ordered_crossover(self, parent1: List[GCodePoint], parent2: List[GCodePoint]) -> List[GCodePoint]:
        """
        有序交叉（保持顺序）
        :param parent1: 父代1
        :param parent2: 父代2
        :return: 子代
        """
        size = len(parent1)
        if size < 2:
            return parent1.copy()
        
        # 选择交叉点
        start, end = sorted(random.sample(range(size), 2))
        
        # 创建子代，保留parent1的中间段
        child = [None] * size
        child[start:end+1] = parent1[start:end+1]
        
        # 使用集合来快速查找已存在的点（使用坐标元组）
        existing = set((p.x, p.y, p.z) for p in child if p is not None)
        
        # 填充剩余位置
        parent2_pos = 0
        for i in range(size):
            if child[i] is None:
                # 找到parent2中不在child中的点
                while (parent2[parent2_pos].x, parent2[parent2_pos].y, parent2[parent2_pos].z) in existing:
                    parent2_pos += 1
                    if parent2_pos >= size:
                        parent2_pos = 0
                
                child[i] = parent2[parent2_pos]
                existing.add((parent2[parent2_pos].x, parent2[parent2_pos].y, parent2[parent2_pos].z))
                parent2_pos += 1
        
        return child
    
    def swap_mutation(self, individual: List[GCodePoint]) -> List[GCodePoint]:
        """
        交换变异
        :param individual: 个体
        :return: 变异后的个体
        """
        size = len(individual)
        if size < 2:
            return individual.copy()
        
        # 随机选择两个位置交换
        pos1, pos2 = random.sample(range(size), 2)
        individual = individual.copy()
        individual[pos1], individual[pos2] = individual[pos2], individual[pos1]
        
        return individual

    def greedy_path_planning(self, raw_points: List[GCodePoint]) -> List[GCodePoint]:
        """
        贪心算法实现路径排序：每次选择适应度最高且无交叉的候选点
        :param raw_points: 未排序的原始离散点
        :return: 规划后的路径点列表
        """
        if not raw_points:
            return []
        
        # 重置路径和访问标记
        self.path = []
        for p in raw_points:
            p.visited = False
        
        # 初始化：选择第一个点作为起点
        current_point = raw_points[0]
        current_point.visited = True
        self.path.append(current_point)
        
        # 进度提示
        total_points = len(raw_points)
        print("\n开始路径规划（避免交叉）...")
        
        # 循环选择最优候选点，直到所有点都被访问
        while len(self.path) < total_points:
            remaining_points = [p for p in raw_points if not p.visited]
            if not remaining_points:
                break
            
            # 计算所有候选点的适应度并排序
            candidate_fitness = []
            for candidate in remaining_points:
                prev_point = self.path[-2] if len(self.path) >= 2 else None
                fitness = self.calculate_fitness(current_point, candidate, prev_point)
                candidate_fitness.append((candidate, fitness))
            
            # 按适应度降序排序
            candidate_fitness.sort(key=lambda x: x[1], reverse=True)
            
            # 寻找最优且无交叉的候选点
            best_candidate = None
            for candidate, fitness in candidate_fitness:
                if not self.is_crossing_path(current_point, candidate):
                    best_candidate = candidate
                    break
            
            # 极端情况：所有候选点都交叉，选择适应度最高的（避免程序卡死）
            if best_candidate is None:
                best_candidate = candidate_fitness[0][0]
                print(f"警告：第{len(self.path)+1}个点无无交叉候选，选择最优适应度点（可能轻微交叉）")
            
            # 将最优候选点加入路径
            best_candidate.visited = True
            self.path.append(best_candidate)
            current_point = best_candidate
            
            # 打印进度
            if len(self.path) % 10 == 0:  # 每10个点打印一次进度
                progress = (len(self.path) / total_points) * 100
                print(f"进度：{len(self.path)}/{total_points} ({progress:.1f}%)")
        
        print(f"路径规划完成！最终路径点数：{len(self.path)}")
        return self.path
    
    def nsga2_path_planning(self, raw_points: List[GCodePoint]) -> List[GCodePoint]:
        """
        NSGA-II算法实现路径排序
        :param raw_points: 未排序的原始离散点
        :return: 规划后的路径点列表
        """
        if not raw_points:
            return []
        
        # 进度提示
        total_points = len(raw_points)
        print("\n开始NSGA-II路径规划...")
        print(f"种群大小: {self.pop_size}, 迭代次数: {self.max_gen}")
        
        # 1. 初始化种群（随机排列的点）
        population = []
        for _ in range(self.pop_size):
            individual = raw_points.copy()
            random.shuffle(individual)  # 随机排列点作为初始个体
            population.append(individual)
        
        # 2. 迭代进化
        for generation in range(self.max_gen):
            # 进度显示
            if (generation + 1) % 10 == 0:
                print(f"  正在进化... 第{generation + 1}/{self.max_gen}代")
            
            # 3. 计算非支配排序和拥挤度
            frontiers = self.non_dominated_sorting(population)
            crowding_distance = self.crowding_distance_calculation(population, frontiers)
            
            # 4. 选择、交叉、变异生成子代
            offspring = []
            while len(offspring) < self.pop_size:
                # 锦标赛选择
                parent1 = self.tournament_selection(population, frontiers, crowding_distance)
                parent2 = self.tournament_selection(population, frontiers, crowding_distance)
                
                # 交叉
                if random.random() < self.crossover_prob:
                    child = self.ordered_crossover(parent1, parent2)
                else:
                    child = parent1.copy()
                
                # 变异
                if random.random() < self.mutation_prob:
                    child = self.swap_mutation(child)
                
                offspring.append(child)
            
            # 5. 合并父代和子代
            combined_population = population + offspring
            
            # 6. 对合并后的种群进行非支配排序和拥挤度计算
            combined_frontiers = self.non_dominated_sorting(combined_population)
            combined_crowding_distance = self.crowding_distance_calculation(combined_population, combined_frontiers)
            
            # 7. 选择下一代种群
            next_population = []
            for frontier in combined_frontiers:
                # 按拥挤度降序排序前沿个体
                frontier_sorted = sorted(frontier, key=lambda i: combined_crowding_distance[i], reverse=True)
                
                # 将前沿个体加入下一代种群，直到满员
                for individual_idx in frontier_sorted:
                    if len(next_population) < self.pop_size:
                        next_population.append(combined_population[individual_idx])
                    else:
                        break
                
                if len(next_population) >= self.pop_size:
                    break
            
            population = next_population
        
        # 3. 选择最优个体
        print("\n进化完成，选择最优个体...")
        frontiers = self.non_dominated_sorting(population)
        crowding_distance = self.crowding_distance_calculation(population, frontiers)
        
        # 选择第一个前沿中拥挤度最高的个体
        best_individual = None
        best_crowding_distance = -float('inf')
        
        for individual_idx in frontiers[0]:
            if crowding_distance[individual_idx] > best_crowding_distance:
                best_crowding_distance = crowding_distance[individual_idx]
                best_individual = population[individual_idx]
        
        # 如果没有找到，选择第一个前沿的第一个个体
        if best_individual is None and frontiers[0]:
            best_individual = population[frontiers[0][0]]
        
        # 设置路径
        self.path = best_individual
        
        print(f"路径规划完成！最终路径点数：{len(self.path)}")
        
        # 计算并显示最优个体的目标值
        objectives = self.calculate_objectives(self.path)
        print(f"\n最优个体目标值：")
        print(f"  总路径长度: {objectives[0]:.2f}")
        print(f"  总曲率惩罚: {objectives[1]:.4f}")
        print(f"  总启停惩罚: {objectives[2]:.4f}")
        print(f"  交叉次数: {objectives[3]}")
        
        return self.path

    def save_planned_gcode(self, output_filename: str = "planned_left_region_gcode.gcode"):
        """将规划后的G代码保存到桌面"""
        desktop_path = self.get_desktop_path()
        output_path = os.path.join(desktop_path, output_filename)
        
        # 生成规划后的G代码
        planned_gcode = [point.cmd for point in self.path]
        
        # 写入文件（添加注释头）
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("; Multi-Head Path Planning Result (No Crossing)\n")
            f.write(f"; theta_th={self.theta_th}°, lambda1={self.lambda1}, lambda2={self.lambda2}\n")
            f.write("; Path constraint: No segment crossing\n")
            f.write("; ==============================\n")
            for line in planned_gcode:
                f.write(line + "\n")
        
        print(f"\n规划后的G代码已保存：{output_path}")

    def generate_planned_gcode(self) -> List[str]:
        """将规划后的路径转换为G代码格式输出"""
        return [point.cmd for point in self.path]

    def visualize_path(self):
        """可视化规划前后的路径对比（2D），标记交叉点（如有）"""
        if not self.raw_points or not self.path:
            print("无数据可可视化！请先执行路径规划")
            return
        
        # 提取原始点和规划路径点的坐标
        raw_x = [p.x for p in self.raw_points]
        raw_y = [p.y for p in self.raw_points]
        
        planned_x = [p.x for p in self.path]
        planned_y = [p.y for p in self.path]
        
        # 检测规划路径中的交叉线段（用于验证）
        crossing_segments = []
        for i in range(len(self.path)-1):
            for j in range(i+2, len(self.path)-1):  # 跳过相邻线段
                seg1_a, seg1_b = self.path[i], self.path[i+1]
                seg2_a, seg2_b = self.path[j], self.path[j+1]
                if self.segments_intersect(seg1_a, seg1_b, seg2_a, seg2_b):
                    crossing_segments.append(((seg1_a, seg1_b), (seg2_a, seg2_b)))
        
        # 创建画布，分为左右两个子图（原始点 vs 规划路径）
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        fig.suptitle("Multi-Head Path Planning Result (No Crossing Constraint)", fontsize=14, fontweight="bold")

        # 子图1：原始离散点（无连接）
        ax1.scatter(raw_x, raw_y, c="red", s=30, label="Raw Points", zorder=2, alpha=0.7)
        ax1.set_title("Original Discrete Points", fontsize=12)
        ax1.set_xlabel("X Coordinate")
        ax1.set_ylabel("Y Coordinate")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_aspect("equal", adjustable="box")  # 等比例显示

        # 子图2：规划后的路径（带连线+锐角标记+交叉标记）
        # 绘制路径连线
        ax2.plot(planned_x, planned_y, c="blue", linewidth=1.5, label="", zorder=1, alpha=0.8)
        # 绘制路径点
        ax2.scatter(planned_x, planned_y, c="green", s=30, label="", zorder=2, alpha=0.7)
        
        # 标记交叉线段（如有）
        if crossing_segments:
            for (seg1, seg2) in crossing_segments:
                # 绘制交叉线段（红色虚线）
                ax2.plot([seg1[0].x, seg1[1].x], [seg1[0].y, seg1[1].y], c="red", linewidth=2, linestyle="--", alpha=0.8)
                ax2.plot([seg2[0].x, seg2[1].x], [seg2[0].y, seg2[1].y], c="red", linewidth=2, linestyle="--", alpha=0.8)
        
        # 标注锐角（<50°）
        sharp_angles = []
        for i in range(1, len(self.path)-1):
            p1 = self.path[i-1]
            p2 = self.path[i]
            p3 = self.path[i+1]
            angle = GCodePoint.calculate_curvature_angle(p1, p2, p3)
            if angle < self.theta_th:
                sharp_angles.append((p2.x, p2.y, angle))
                # 在锐角点绘制红色五角星标记
                ax2.scatter(p2.x, p2.y, c="red", marker="*", s=120, zorder=3)
                # 标注角度值（仅标注前20个锐角，避免图面拥挤）
                if len(sharp_angles) <= 20:
                    ax2.text(p2.x+0.5, p2.y-0.5, f"{angle:.1f}°", fontsize=7, color="red", fontweight="bold")

        ax2.set_title(f"", fontsize=12)
        ax2.set_xlabel("X Coordinate")
        ax2.set_ylabel("Y Coordinate")
        ax2.grid(True, alpha=0.3)
        
        # 更新图例（添加交叉标记）
        legend_elements = [
            plt.Line2D([0], [0], color='blue', lw=1.5, label=''),
            plt.scatter([0], [0], c='green', s=30, label=''),
            plt.scatter([0], [0], c='red', marker='*', s=120, label=''),
        ]
        if crossing_segments:
            legend_elements.append(plt.Line2D([0], [0], color='red', lw=2, linestyle='--', label=''))
        ax2.legend(handles=legend_elements)
        
        ax2.set_aspect("equal", adjustable="box")  # 等比例显示

        # 标注统计信息
        stats_text = f"""

        """.strip()
        ax2.text(0.02, 0.98, stats_text, 
                 transform=ax2.transAxes, fontsize=9, color="black", 
                 verticalalignment="top", bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8))

        # 调整布局并显示
        plt.tight_layout()
        plt.show()

# ---------------------- 主程序 ----------------------
if __name__ == "__main__":
    try:
        # 1. 初始化规划器
        planner = MultiHeadPathPlanner(theta_th=50.0, lambda1=0.4, lambda2=0.3)

        # 2. 读取桌面的G代码文件
        gcode_lines = planner.read_gcode_file("right_region_gcode.gcode")

        # 3. 解析G代码提取坐标点
        raw_points = planner.parse_gcode(gcode_lines)

        # 4. 执行路径规划（使用NSGA-II算法）
        planned_points = planner.nsga2_path_planning(raw_points)

        # 5. 输出规划后的G代码（控制台）
        print("\n" + "="*60)
        print("规划后的G代码（前20行）：")
        planned_gcode = planner.generate_planned_gcode()
        for i, line in enumerate(planned_gcode[:20]):
            print(f"{i+1:3d}: {line}")
        if len(planned_gcode) > 20:
            print(f"      ... 共{len(planned_gcode)}行")

        # 6. 保存规划后的G代码到桌面
        planner.save_planned_gcode("planned_left_region_gcode_nsga2.gcode")

        # 7. 可视化路径
        planner.visualize_path()

    except Exception as e:
        print(f"程序执行出错：{e}")
        import traceback
        traceback.print_exc()