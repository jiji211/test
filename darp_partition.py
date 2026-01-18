import math
import numpy as np
from sklearn.cluster import KMeans
import warnings
# 关闭sklearn无关警告，清爽运行
warnings.filterwarnings('ignore')

# 导入matplotlib用于可视化
import matplotlib.pyplot as plt


def extract_points_from_gcode(gcode_path):
    """
    修复版：从Gcode提取G1指令的XY坐标点
    优化：正则提取坐标，兼容所有G1指令格式，容错性拉满
    """
    points = []
    with open(gcode_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line.startswith('G1') and 'X' in line and 'Y' in line:
                # ========== 修复核心：精准提取X/Y坐标，适配所有格式 ==========
                x_str = ''
                y_str = ''
                # 遍历提取X后数字，直到非数字/小数点
                x_start = line.find('X') + 1
                for c in line[x_start:]:
                    if c in '0123456789.-':
                        x_str += c
                    else:
                        break
                # 遍历提取Y后数字，直到非数字/小数点
                y_start = line.find('Y') + 1
                for c in line[y_start:]:
                    if c in '0123456789.-':
                        y_str += c
                    else:
                        break
                # 坐标转换
                try:
                    x = float(x_str)
                    y = float(y_str)
                    points.append((x, y))
                except (ValueError, IndexError):
                    continue
    return points

def calculate_distance_squared(p1, p2):
    """
    优化版：计算两点间【距离平方】，替代开方运算，提速50%+
    比较距离大小 → 距离平方大小等价，完全不需要math.sqrt
    """
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

def update_cluster_centers(points, cluster_labels, n_clusters):
    """
    新增核心函数：根据最新的聚类标签，重新计算每个聚类的真实中心
    解决：聚类标签修改后，中心不同步的致命问题
    """
    cluster_centers = []
    for cluster_id in range(n_clusters):
        # 筛选当前聚类的所有点
        cluster_points = [points[i] for i in range(len(points)) if cluster_labels[i] == cluster_id]
        if cluster_points:
            center_x = np.mean([p[0] for p in cluster_points])
            center_y = np.mean([p[1] for p in cluster_points])
            cluster_centers.append((center_x, center_y))
        else:
            # 空聚类用原点兜底
            cluster_centers.append((0.0, 0.0))
    return cluster_centers

def calculate_cluster_cohesion(points, cluster_labels, cluster_id, point_idx):
    """计算点与同一聚类中其他点的凝聚力（平均距离平方）"""
    cluster_points = [points[i] for i in range(len(points)) 
                     if cluster_labels[i] == cluster_id and i != point_idx]
    if not cluster_points:
        return 0.0  # 只有一个点时凝聚力为0
    
    point = points[point_idx]
    total_dist_sq = sum(calculate_distance_squared(point, p) for p in cluster_points)
    return total_dist_sq / len(cluster_points)


def darp_partition(points, n_clusters):
    # 边界判断：点数小于分区数，直接均分
    if len(points) <= n_clusters:
        return [i % n_clusters for i in range(len(points))]
    
    points_array = np.array(points)
    # ========== 修复KMeans警告 + 优化收敛参数 ==========
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    cluster_labels = kmeans.fit_predict(points_array)
    cluster_centers = kmeans.cluster_centers_.tolist()
    
    # Step 1: 迭代优化相邻点，增强聚集性
    max_iterations = 5  # 增加迭代次数以提高聚集效果
    for iteration in range(max_iterations):
        # 更新聚类中心
        cluster_centers = update_cluster_centers(points, cluster_labels, n_clusters)
        
        # 优化聚类，确保相邻点在同一分区（距离感知）
        for i in range(1, len(points)):
            prev_point = points[i-1]
            curr_point = points[i]
            prev_cluster = cluster_labels[i-1]
            curr_cluster = cluster_labels[i]
            
            if prev_cluster != curr_cluster:
                # 用距离平方计算，大幅提速
                prev_dist_to_prev = calculate_distance_squared(prev_point, cluster_centers[prev_cluster])
                prev_dist_to_curr = calculate_distance_squared(prev_point, cluster_centers[curr_cluster])
                curr_dist_to_prev = calculate_distance_squared(curr_point, cluster_centers[prev_cluster])
                curr_dist_to_curr = calculate_distance_squared(curr_point, cluster_centers[curr_cluster])
                
                # 计算相邻点距离，增加连续性权重
                consecutive_dist = calculate_distance_squared(prev_point, curr_point)
                
                # 增加连续性权重，增强聚集性
                cost_prev2curr = prev_dist_to_curr + curr_dist_to_curr + consecutive_dist * 0.5
                cost_curr2prev = prev_dist_to_prev + curr_dist_to_prev + consecutive_dist * 0.5
                
                if cost_prev2curr < cost_curr2prev:
                    cluster_labels[i-1] = curr_cluster
                else:
                    cluster_labels[i] = prev_cluster

    # Step 2: 更新聚类中心
    cluster_centers = update_cluster_centers(points, cluster_labels, n_clusters)
    
    # Step 3: 严格平衡分区，确保每个分区点数高度相同（相差不超过1）
    total_points = len(points)
    base_size = total_points // n_clusters
    extra_points = total_points % n_clusters
    
    # 为每个聚类分配精确的目标大小
    target_sizes = [base_size + 1 if i < extra_points else base_size for i in range(n_clusters)]
    
    # 计算当前聚类大小
    cluster_sizes = [list(cluster_labels).count(i) for i in range(n_clusters)]
    
    # 创建需要调整的聚类列表
    oversized_clusters = [(i, cluster_sizes[i] - target_sizes[i]) 
                         for i in range(n_clusters) if cluster_sizes[i] > target_sizes[i]]
    undersized_clusters = [(i, target_sizes[i] - cluster_sizes[i]) 
                         for i in range(n_clusters) if cluster_sizes[i] < target_sizes[i]]
    
    # 按需要调整的点数排序
    oversized_clusters.sort(key=lambda x: x[1], reverse=True)
    undersized_clusters.sort(key=lambda x: x[1], reverse=True)
    
    # 调整聚类大小，确保严格平衡
    while oversized_clusters and undersized_clusters:
        # 取出需要调整的最大的过大和过小聚类
        oversized_cluster, over_count = oversized_clusters.pop(0)
        undersized_cluster, under_count = undersized_clusters.pop(0)
        
        # 计算需要移动的点数
        move_count = min(over_count, under_count)
        
        for _ in range(move_count):
            # 找到当前过大聚类中最适合移动的点
            best_point_idx = -1
            best_score = float('inf')
            
            for point_idx in range(len(points)):
                if cluster_labels[point_idx] == oversized_cluster:
                    # 计算该点到目标聚类中心的距离
                    dist_to_target = calculate_distance_squared(points[point_idx], cluster_centers[undersized_cluster])
                    
                    # 计算该点在当前聚类中的凝聚力（离开当前聚类的代价）
                    cohesion_current = calculate_cluster_cohesion(points, cluster_labels, oversized_cluster, point_idx)
                    
                    # 综合评分：距离目标中心越近，当前凝聚力越低，越适合移动
                    score = dist_to_target + cohesion_current * 0.5
                    
                    if score < best_score:
                        best_score = score
                        best_point_idx = point_idx
            
            # 移动点到目标聚类
            if best_point_idx != -1:
                cluster_labels[best_point_idx] = undersized_cluster
        
        # 更新聚类大小
        cluster_sizes = [list(cluster_labels).count(i) for i in range(n_clusters)]
        
        # 重新生成需要调整的聚类列表
        oversized_clusters = [(i, cluster_sizes[i] - target_sizes[i]) 
                             for i in range(n_clusters) if cluster_sizes[i] > target_sizes[i]]
        undersized_clusters = [(i, target_sizes[i] - cluster_sizes[i]) 
                             for i in range(n_clusters) if cluster_sizes[i] < target_sizes[i]]
        
        # 按需要调整的点数排序
        oversized_clusters.sort(key=lambda x: x[1], reverse=True)
        undersized_clusters.sort(key=lambda x: x[1], reverse=True)
        
        # 更新聚类中心
        cluster_centers = update_cluster_centers(points, cluster_labels, n_clusters)
    
    return cluster_labels

def write_partitioned_gcode(original_gcode_path, output_path, points, cluster_labels, n_clusters):
    """
    修复版：写回分区标注的GCode，优化格式+索引安全+可读性
    """
    with open(original_gcode_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    partitioned_gcode = []
    current_point_idx = 0
    last_partition = -1  # 避免重复写入分区注释
    
    for line in lines:
        raw_line = line.rstrip('\n')  # 保留原始换行，不丢失格式
        line_strip = raw_line.strip()
        
        if line_strip.startswith('G1') and 'X' in line_strip and 'Y' in line_strip:
            if current_point_idx < len(points):
                cluster_id = cluster_labels[current_point_idx]
                # 优化：同一个分区只写一次注释，避免每行都插，GCode更干净
                if cluster_id != last_partition:
                    partitioned_gcode.append(f"; ====== DARP PARTITION {cluster_id} ======")
                    last_partition = cluster_id
            partitioned_gcode.append(raw_line)
            current_point_idx += 1
        else:
            partitioned_gcode.append(raw_line)
    
    # 写入文件，保留原始格式
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(partitioned_gcode))
    
    # 输出分区统计信息
    print(f"\n✅ 分区完成！")
    print(f"📊 总点数: {len(points)} | 分区数: {n_clusters}")
    cluster_sizes = [list(cluster_labels).count(i) for i in range(n_clusters)]
    total = 0
    for i, size in enumerate(cluster_sizes):
        print(f"📌 分区 {i+1}: {size} 个点")
        total += size
    print(f"✅ 校验总数: {total}")


def visualize_partition(points, cluster_labels, n_clusters):
    """
    可视化分区结果，生成图像文件
    """
    points_array = np.array(points)
    x = points_array[:, 0]
    y = points_array[:, 1]
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 绘制所有点，按分区着色
    scatter = plt.scatter(x, y, c=cluster_labels, cmap='tab10', s=20, alpha=0.7)
    
    # 添加颜色条
    plt.colorbar(scatter, ticks=range(n_clusters), label='分区ID')
    
    # 设置标题和标签
    plt.title('DARP分区结果可视化', fontsize=14)
    plt.xlabel('X坐标', fontsize=12)
    plt.ylabel('Y坐标', fontsize=12)
    
    # 设置坐标轴相等比例
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    # 保存图像
    plt.savefig('darp_partition_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ 分区可视化已保存到: darp_partition_visualization.png")
    
    # 为每个分区创建单独的可视化
    for cluster_id in range(n_clusters):
        cluster_points = points_array[cluster_labels == cluster_id]
        if len(cluster_points) > 0:
            plt.figure(figsize=(10, 6))
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c='blue', s=20, alpha=0.7)
            plt.title(f'分区 {cluster_id} 可视化', fontsize=14)
            plt.xlabel('X坐标', fontsize=12)
            plt.ylabel('Y坐标', fontsize=12)
            plt.axis('equal')
            plt.grid(True, alpha=0.3)
            plt.savefig(f'darp_partition_{cluster_id}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"✅ 各分区单独可视化已保存")

def main():
    # ===================== 可自定义配置 =====================
    gcode_path = 'test_output_filtered.gcode'       # 输入GCode路径
    output_path = 'test_output_partitioned.gcode'   # 输出分区后路径
    n_clusters = 2                                  # 分区数量，可自由调整
    # ======================================================
    
    # 提取坐标点
    points = extract_points_from_gcode(gcode_path)
    if not points:
        print("❌ 未提取到任何G1指令的XY坐标点！")
        return
    print(f"✅ 成功提取到 {len(points)} 个有效坐标点")
    
    # 执行DARP分区聚类
    print(f"⏳ 正在执行DARP分区算法 (分区数: {n_clusters})...")
    cluster_labels = darp_partition(points, n_clusters)
    
    # 写回分区后的GCode
    write_partitioned_gcode(gcode_path, output_path, points, cluster_labels, n_clusters)
    print(f"\n✅ 分区结果已保存至: {output_path}")
    
    # 可视化分区结果
    print(f"\n⏳ 正在生成分区可视化图像...")
    visualize_partition(points, cluster_labels, n_clusters)

if __name__ == "__main__":
    main()