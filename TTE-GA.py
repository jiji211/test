import math
import numpy as np
import random
import matplotlib.pyplot as plt
import logging

# 配置日志记录
logging.basicConfig(
    filename='genetic_algorithm.log',
    level=logging.DEBUG,  # 使用DEBUG级别以获取更详细的信息
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 设置随机种子以确保结果可重复
random.seed(42)
np.random.seed(42)

class GeneticPathPlanner:
    def __init__(self, partitioned_gcode_path):
        self.partitioned_gcode_path = partitioned_gcode_path
        self.points = []  # 所有点的坐标
        self.partitions = []  # 每个点的分区标签
        self.partition_points = {}  # 按分区组织的点
        self.current_partition = -1
        
        # 算法参数
        self.pop_size = 20  # 减少种群大小以提高性能
        self.cross_rate = 0.8
        self.mutate_rate = 0.1
        self.λ1 = 30
        self.λ2 = 1
        
        self.extract_partitioned_points()
        self.organize_points_by_partition()
    
    def extract_partitioned_points(self):
        """从分区后的G代码中提取点和分区信息"""
        with open(self.partitioned_gcode_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                
                # 提取分区信息
                if line.startswith('; ====== DARP PARTITION'):
                    try:
                        self.current_partition = int(line.split('DARP PARTITION')[1].split('======')[0].strip())
                    except (ValueError, IndexError):
                        continue
                
                # 提取G1指令的坐标
                elif line.startswith('G1') and 'X' in line and 'Y' in line:
                    # 提取X坐标
                    x_start = line.find('X') + 1
                    x_end = x_start
                    while x_end < len(line) and (line[x_end].isdigit() or line[x_end] in '.-'):
                        x_end += 1
                    x_str = line[x_start:x_end]
                    
                    # 提取Y坐标
                    y_start = line.find('Y') + 1
                    y_end = y_start
                    while y_end < len(line) and (line[y_end].isdigit() or line[y_end] in '.-'):
                        y_end += 1
                    y_str = line[y_start:y_end]
                    
                    try:
                        x = float(x_str)
                        y = float(y_str)
                        self.points.append((x, y))
                        self.partitions.append(self.current_partition)
                    except (ValueError, IndexError):
                        continue
    
    def organize_points_by_partition(self):
        """将点按分区组织"""
        for i, (point, partition) in enumerate(zip(self.points, self.partitions)):
            if partition not in self.partition_points:
                self.partition_points[partition] = []
            self.partition_points[partition].append((i, point))
    
    def calculate_distance(self, p1, p2):
        """计算两点之间的欧几里得距离"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def segments_intersect(self, a1, a2, b1, b2):
        """检测两条线段是否相交"""
        def ccw(p, q, r):
            return (q[1] - p[1]) * (r[0] - q[0]) > (q[0] - p[0]) * (r[1] - q[1])
        
        return ccw(a1, b1, b2) != ccw(a2, b1, b2) and ccw(a1, a2, b1) != ccw(a1, a2, b2)
    
    def path_cross_count(self, individual):
        """计算路径中的交叉次数 - 完整检查所有线段对"""
        cross_count = 0
        path = [self.points[idx] for idx in individual]
        n = len(path)
        
        for i in range(n - 1):
            # 检查所有后续不相邻的线段对
            for j in range(i + 2, n - 1):
                seg1_a = path[i]
                seg1_b = path[i + 1]
                seg2_a = path[j]
                seg2_b = path[j + 1]
                
                # 快速排除不相交的线段（边界检查）
                if (max(seg1_a[0], seg1_b[0]) < min(seg2_a[0], seg2_b[0]) or 
                    max(seg2_a[0], seg2_b[0]) < min(seg1_a[0], seg1_b[0]) or
                    max(seg1_a[1], seg1_b[1]) < min(seg2_a[1], seg2_b[1]) or 
                    max(seg2_a[1], seg2_b[1]) < min(seg1_a[1], seg1_b[1])):
                    continue
                
                # 详细交叉检测
                if self.segments_intersect(seg1_a, seg1_b, seg2_a, seg2_b):
                    cross_count += 1
        
        return cross_count
    
    def calculate_fitness(self, individual):

        total_distance = 0.0
        
        # 计算路径总长度
        for i in range(len(individual) - 1):
            point1 = self.points[individual[i]]
            point2 = self.points[individual[i+1]]
            total_distance += self.calculate_distance(point1, point2)
        
        # 计算交叉次数
        cross_count = self.path_cross_count(individual)
        
        # 强制约束：如果有交叉点，适应度为负无穷
        if cross_count > 0:
            return -float('inf')
        
        # 只有无交叉的路径才计算适应度，距离越短适应度越高
        fitness = 1.0 / (total_distance + 1e-6)
        
        return fitness
    
    def roulette_wheel_selection(self, population, fitness_list):
        """轮盘赌选择"""
        total_fitness = sum(fitness_list)
        probabilities = [f / total_fitness for f in fitness_list]
        
        # 选择第一个父代
        r1 = random.random()
        cumulative_prob = 0.0
        parent1_idx = 0
        for i, prob in enumerate(probabilities):
            cumulative_prob += prob
            if cumulative_prob >= r1:
                parent1_idx = i
                break
        
        # 选择第二个父代（确保不与第一个相同）
        r2 = random.random()
        cumulative_prob = 0.0
        parent2_idx = 0
        for i, prob in enumerate(probabilities):
            cumulative_prob += prob
            if cumulative_prob >= r2 and i != parent1_idx:
                parent2_idx = i
                break
        
        return population[parent1_idx], population[parent2_idx]
    
    def pmx_crossover(self, parent1, parent2, crossover_point):
        """部分映射交叉"""
        size = len(parent1)
        offspring = [-1] * size
        
        # 复制父代1的前半部分到子代
        offspring[:crossover_point] = parent1[:crossover_point]
        
        # 处理父代2的映射
        mapping = {}
        for i in range(crossover_point, size):
            if parent2[i] not in offspring:
                offspring[i] = parent2[i]
            else:
                # 查找映射
                mapped_gene = parent1[parent2.index(parent2[i])]
                while mapped_gene in offspring:
                    mapped_gene = parent1[parent2.index(mapped_gene)]
                offspring[i] = mapped_gene
        
        return offspring
    
    def region_assignment_algorithm(self, offspring, a_set):
        """区域分配算法 - 简化实现"""
        return offspring
    
    def generate_initial_population(self):
        """生成初始种群 - 使用简单的非交叉路径生成算法"""
        logging.debug("Generating initial population with non-crossing paths...")
        population = []
        
        # 获取所有分区ID
        partitions = sorted(self.partition_points.keys())
        
        for _ in range(self.pop_size):
            individual = []
            
            # 对每个分区的点进行顺序排列，避免交叉
            for partition in partitions:
                # 获取该分区的所有点索引
                partition_indices = [idx for idx, _ in self.partition_points[partition]]
                
                # 根据点的x坐标排序，生成简单的无交叉路径
                # 对于x坐标相近的点，使用y坐标排序
                partition_indices.sort(key=lambda idx: (self.points[idx][0], self.points[idx][1]))
                
                # 随机决定是正序还是逆序，增加多样性
                if random.random() > 0.5:
                    partition_indices.reverse()
                
                individual.extend(partition_indices)
            
            population.append(individual)
        
        logging.debug(f"Generated {len(population)} individuals with simple non-crossing paths")
        return population
    
    def mutate(self, individual):
        """变异操作 - 交换两个点的位置"""
        pos1 = random.randint(0, len(individual) - 1)
        pos2 = random.randint(0, len(individual) - 1)
        
        while pos1 == pos2:
            pos2 = random.randint(0, len(individual) - 1)
        
        # 交换位置
        individual[pos1], individual[pos2] = individual[pos2], individual[pos1]
        return individual
    
    def remove_crossings(self, individual):
        path = [self.points[idx] for idx in individual]
        n = len(path)
        
        # 检测并修复交叉
        while True:
            crossing_found = False
            
            for i in range(n - 1):
                for j in range(i + 2, n - 1):  # 跳过相邻线段
                    seg1_a = path[i]
                    seg1_b = path[i + 1]
                    seg2_a = path[j]
                    seg2_b = path[j + 1]
                    
                    if self.segments_intersect(seg1_a, seg1_b, seg2_a, seg2_b):
                        # 交换第二个线段的两个端点来移除交叉
                        path[j], path[j + 1] = path[j + 1], path[j]
                        individual[j], individual[j + 1] = individual[j + 1], individual[j]
                        crossing_found = True
                        # 重新检查整个路径，确保没有新的交叉点产生
                        break
                if crossing_found:
                    break
            
            if not crossing_found:
                break  # 没有找到交叉点，退出循环
        
        return individual
    
    def run_genetic_algorithm(self, generations=100):
        """运行遗传算法"""
        logging.info("Starting genetic algorithm...")
        
        # 生成初始种群（已确保无交叉）
        population = self.generate_initial_population()
        logging.info(f"Generated initial population with {len(population)} valid non-crossing individuals")
        
        best_fitness = -float('inf')
        best_individual = None
        best_cross_count = float('inf')
        
        print(f"Starting Genetic Algorithm with {self.pop_size} individuals, {generations} generations")
        print(f"Cross penalty weight: {50000.0}")
        
        try:
            for generation in range(generations):
                logging.debug(f"Processing generation {generation + 1}/{generations}")
                
                # 计算适应度
                fitness_list = []
                cross_counts = []
                
                for i, individual in enumerate(population):
                    if i % 10 == 0:  # 每10个个体记录一次日志，避免日志过多
                        logging.debug(f"Calculating fitness for individual {i}/{self.pop_size}")
                    
                    fitness = self.calculate_fitness(individual)
                    fitness_list.append(fitness)
                    cross_count = self.path_cross_count(individual)
                    cross_counts.append(cross_count)
                
                # 更新最佳个体
                current_best_idx = np.argmax(fitness_list)
                current_best_fitness = fitness_list[current_best_idx]
                current_best_cross = cross_counts[current_best_idx]
                
                if current_best_fitness > best_fitness:
                    best_fitness = current_best_fitness
                    best_individual = population[current_best_idx].copy()
                    best_cross_count = current_best_cross
                    logging.info(f"New best individual found at generation {generation + 1}: Fitness = {best_fitness:.6f}, Cross Count = {best_cross_count}")
                
                # 生成新一代
                new_population = []
                logging.debug(f"Generating new population...")
                
                while len(new_population) < self.pop_size:
                    # 选择父母
                    parent1, parent2 = self.roulette_wheel_selection(population, fitness_list)
                    
                    # 创建子代
                    offspring = parent1.copy()
                    
                    # 交叉操作
                    if random.random() < self.cross_rate:
                        crossover_point = random.randint(1, len(parent1) - 1)
                        offspring = self.pmx_crossover(parent1, parent2, crossover_point)
                    
                    # 变异操作
                    if random.random() < self.mutate_rate:
                        offspring = self.mutate(offspring)
                    
                    # 移除交叉点，确保路径无交叉
                    offspring = self.remove_crossings(offspring)
                    
                    # 检查是否需要区域分配
                    if offspring == parent1 == parent2:
                        offspring = self.region_assignment_algorithm(offspring, None)
                    
                    new_population.append(offspring)
                
                population = new_population
                
                # 每10代打印一次进度
                if (generation + 1) % 10 == 0:
                    avg_cross = sum(cross_counts) / len(cross_counts)
                    current_avg_fitness = sum(fitness_list) / len(fitness_list)
                    print(f"Generation {generation + 1}: Best Fitness = {best_fitness:.6f}, Best Cross Count = {best_cross_count}, Avg Fitness = {current_avg_fitness:.6f}, Avg Cross Count = {avg_cross:.2f}")
                    logging.info(f"Generation {generation + 1}: Best Fitness = {best_fitness:.6f}, Best Cross Count = {best_cross_count}, Avg Fitness = {current_avg_fitness:.6f}, Avg Cross Count = {avg_cross:.2f}")
        
        except Exception as e:
            logging.error(f"Error in genetic algorithm: {str(e)}", exc_info=True)
            raise
        
        # 最终结果
        final_cross_count = self.path_cross_count(best_individual)
        final_distance = 0.0
        for i in range(len(best_individual) - 1):
            point1 = self.points[best_individual[i]]
            point2 = self.points[best_individual[i+1]]
            final_distance += self.calculate_distance(point1, point2)
        
        print(f"\nFinal Results:")
        print(f"  Best Fitness: {best_fitness:.6f}")
        print(f"  Path Length: {final_distance:.2f}")
        print(f"  Cross Count: {final_cross_count}")
        print(f"  Path Points: {len(best_individual)}")
        
        logging.info(f"Genetic algorithm completed successfully")
        logging.info(f"Final Results: Best Fitness = {best_fitness:.6f}, Path Length = {final_distance:.2f}, Cross Count = {final_cross_count}, Path Points = {len(best_individual)}")
        
        return best_individual
    
    def generate_path_gcode(self, best_individual, output_path):
        """生成路径G代码"""
        # 读取原始G代码的头部信息（非G1指令）
        header = []
        with open(self.partitioned_gcode_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.rstrip('\n')
                if not (line.strip().startswith('G1') and 'X' in line and 'Y' in line):
                    header.append(line)
                else:
                    break
        
        # 生成路径G代码
        path_gcode = header
        
        # 添加路径点
        for idx in best_individual:
            x, y = self.points[idx]
            path_gcode.append(f"G1 X{x:.2f} Y{y:.2f}")
        
        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(path_gcode))
    
    def visualize_path(self, best_individual, output_image_path):
        # 提取路径点
        path_points = [self.points[idx] for idx in best_individual]
        x = [p[0] for p in path_points]
        y = [p[1] for p in path_points]
        
        plt.figure(figsize=(12, 8))
        
        # 绘制所有点
        all_x = [p[0] for p in self.points]
        all_y = [p[1] for p in self.points]
        plt.scatter(all_x, all_y, c=self.partitions, cmap='tab10', s=20, alpha=0.5, label='All Points')
        
        # 检测并标记交叉点
        intersections = []
        for i in range(len(path_points) - 1):
            for j in range(i + 2, len(path_points) - 1):  # 跳过相邻线段
                seg1_a = path_points[i]
                seg1_b = path_points[i + 1]
                seg2_a = path_points[j]
                seg2_b = path_points[j + 1]
                
                if self.segments_intersect(seg1_a, seg1_b, seg2_a, seg2_b):
                    # 计算交点（简化为线段中点）
                    int_x = (seg1_a[0] + seg1_b[0] + seg2_a[0] + seg2_b[0]) / 4
                    int_y = (seg1_a[1] + seg1_b[1] + seg2_a[1] + seg2_b[1]) / 4
                    intersections.append((int_x, int_y))
        
        # 绘制路径
        plt.plot(x, y, '-', linewidth=1, color='red', alpha=0.7, label='Path')
        
        # 标记交叉点
        if intersections:
            int_x, int_y = zip(*intersections)
            plt.scatter(int_x, int_y, s=100, color='purple', marker='x', label=f'Intersections ({len(intersections)})')
        
        # 添加起点和终点标记
        plt.scatter(x[0], y[0], s=100, color='green', marker='o', label='Start')
        plt.scatter(x[-1], y[-1], s=100, color='blue', marker='x', label='End')
        
        plt.title('Genetic Algorithm Path Planning Result', fontsize=14)
        plt.xlabel('X Coordinate', fontsize=12)
        plt.ylabel('Y Coordinate', fontsize=12)
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
        plt.close()


def main():
    # 输入和输出文件路径
    partitioned_gcode_path = 'test_output_partitioned.gcode'
    output_gcode_path = 'optimized_path.gcode'
    output_image_path = 'path_visualization.png'
    
    logging.info("Starting Genetic Path Planner...")
    print("Initializing Genetic Path Planner...")
    
    try:
        planner = GeneticPathPlanner(partitioned_gcode_path)
        logging.info(f"Loaded {len(planner.points)} points in {len(planner.partition_points)} partitions")
        print(f"Found {len(planner.points)} points in {len(planner.partition_points)} partitions")
        
        # 运行遗传算法
        print("Running Genetic Algorithm...")
        best_individual = planner.run_genetic_algorithm(generations=50)
        logging.info(f"Genetic algorithm completed. Best individual length: {len(best_individual)}")
        
        # 生成优化后的G代码
        print("Generating Optimized G-code...")
        planner.generate_path_gcode(best_individual, output_gcode_path)
        logging.info(f"Optimized G-code saved to: {output_gcode_path}")
        
        # 可视化路径
        print("Visualizing Path...")
        planner.visualize_path(best_individual, output_image_path)
        logging.info(f"Path visualization saved to: {output_image_path}")
        
        print(f"\n✅ Path planning completed!")
        print(f"   - Optimized G-code saved to: {output_gcode_path}")
        print(f"   - Path visualization saved to: {output_image_path}")
        
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}", exc_info=True)
        print(f"\n❌ Error occurred: {str(e)}")


if __name__ == "__main__":
    main()
