import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
from stl import mesh

class STLSlicer:
    def __init__(self, stl_file):
        self.stl_file = stl_file
        self.mesh = None
        self.load_stl()
        self.scale_factor = 25.0  # 放大100倍
        self.discretize_interval = 30.0  # 40mm间距
        self.filter_distance = 20.0  # 过滤距离阈值（20mm）

    def load_stl(self):
        """使用numpy-stl库加载STL文件"""
        try:
            self.mesh = mesh.Mesh.from_file(self.stl_file)
            print(f"STL文件加载成功: {self.stl_file}")
            print(f"三角面片数量: {len(self.mesh)}")
        except Exception as e:
            print(f"加载STL文件失败: {e}")
            raise

    def slice_mesh(self, z_height):
        """在指定z高度对模型进行切片，返回切片轮廓线段"""
        slices = []
        
        for i in range(len(self.mesh)):
            # 获取三角面片的三个顶点
            v1 = self.mesh.vectors[i][0]
            v2 = self.mesh.vectors[i][1]
            v3 = self.mesh.vectors[i][2]
            
            # 计算三角面片与切片平面的交点
            intersect_points = self.plane_triangle_intersection(v1, v2, v3, z_height)
            
            if len(intersect_points) == 2:
                # 缩放坐标
                scaled_p1 = (intersect_points[0][0] * self.scale_factor, intersect_points[0][1] * self.scale_factor)
                scaled_p2 = (intersect_points[1][0] * self.scale_factor, intersect_points[1][1] * self.scale_factor)
                slices.append([scaled_p1, scaled_p2])
        
        return slices

    def plane_triangle_intersection(self, v1, v2, v3, z_height):
        """计算平面与三角形的交点"""
        intersect_points = []
        
        # 检查三个顶点是否在平面上
        if np.isclose(v1[2], z_height):
            intersect_points.append(v1)
        if np.isclose(v2[2], z_height):
            intersect_points.append(v2)
        if np.isclose(v3[2], z_height):
            intersect_points.append(v3)
            
        # 如果有1个或3个点在平面上，返回这些点
        if len(intersect_points) == 1:
            return intersect_points
        if len(intersect_points) == 3:
            return [intersect_points[0], intersect_points[1]]
        
        # 检查三角形的三条边与平面的交点
        edges = [(v1, v2), (v2, v3), (v3, v1)]
        
        for edge in edges:
            p1, p2 = edge
            
            # 检查边是否与平面相交
            if (p1[2] - z_height) * (p2[2] - z_height) < 0:
                # 计算交点
                t = (z_height - p1[2]) / (p2[2] - p1[2])
                x = p1[0] + t * (p2[0] - p1[0])
                y = p1[1] + t * (p2[1] - p1[1])
                z = z_height
                intersect_points.append(np.array([x, y, z]))
        
        return intersect_points

    def reconstruct_contours(self, segments):
        """将离散线段重建为闭合或开放的轮廓"""
        if not segments:
            return []
            
        contours = []
        used_segments = set()
        
        for i, segment in enumerate(segments):
            if i in used_segments:
                continue
                
            contour = [segment[0], segment[1]]
            used_segments.add(i)
            
            # 尝试扩展轮廓
            while True:
                extended = False
                for j, seg in enumerate(segments):
                    if j in used_segments:
                        continue
                        
                    # 检查当前轮廓的最后一个点是否与线段的起点相连
                    if np.allclose(contour[-1], seg[0], atol=1e-3):
                        contour.append(seg[1])
                        used_segments.add(j)
                        extended = True
                        break
                    # 检查当前轮廓的最后一个点是否与线段的终点相连
                    elif np.allclose(contour[-1], seg[1], atol=1e-3):
                        contour.append(seg[0])
                        used_segments.add(j)
                        extended = True
                        break
                
                if not extended:
                    break
            
            # 检查轮廓是否闭合
            if len(contour) > 2 and np.allclose(contour[0], contour[-1], atol=1e-3):
                # 闭合轮廓，移除最后一个点（与第一个点相同）
                contour.pop()
            
            # 检查轮廓是否已经存在（避免重复轮廓）
            duplicate = False
            
            # 计算当前轮廓的基本特征
            contour_array = np.array(contour)
            if len(contour_array) < 3:
                continue
                
            # 计算轮廓的面积和周长
            current_area = 0.5 * abs(np.sum(contour_array[:-1, 0] * contour_array[1:, 1] - contour_array[1:, 0] * contour_array[:-1, 1]))
            current_perimeter = np.sum(np.linalg.norm(np.diff(contour_array, axis=0), axis=1))
            
            for existing_contour in contours:
                existing_array = np.array(existing_contour)
                if len(existing_array) < 3:
                    continue
                    
                # 计算现有轮廓的面积和周长
                existing_area = 0.5 * abs(np.sum(existing_array[:-1, 0] * existing_array[1:, 1] - existing_array[1:, 0] * existing_array[:-1, 1]))
                existing_perimeter = np.sum(np.linalg.norm(np.diff(existing_array, axis=0), axis=1))
                
                # 如果面积或周长差异太大，不可能是重复轮廓
                if abs(current_area - existing_area) > 10.0 or abs(current_perimeter - existing_perimeter) > 10.0:
                    continue
                    
                # 检查所有点是否相同（允许顺序不同，因为轮廓是闭合的）
                match = False
                # 尝试不同的起始点匹配
                for offset in range(len(existing_contour)):
                    if all(np.allclose(contour[i], existing_contour[(i+offset)%len(existing_contour)], atol=1e-3) for i in range(len(contour))):
                        match = True
                        break
                if match:
                    duplicate = True
                    break
                
                # 检查是否是相似轮廓（处理点数量略有不同的情况）
                if len(contour) > 5 and len(existing_contour) > 5:
                    # 取轮廓的关键点（起点、中间点、终点）进行比较
                    key_points_contour = [
                        contour[0], 
                        contour[len(contour)//2], 
                        contour[-1]
                    ]
                    for offset in range(len(existing_contour)):
                        key_points_existing = [
                            existing_contour[offset],
                            existing_contour[(offset + len(existing_contour)//2) % len(existing_contour)],
                            existing_contour[(offset + len(existing_contour) - 1) % len(existing_contour)]
                        ]
                        if all(np.allclose(p1, p2, atol=5.0) for p1, p2 in zip(key_points_contour, key_points_existing)):
                            match = True
                            break
                    if match:
                        duplicate = True
                        break
            
            if not duplicate:
                contours.append(contour)
        
        return contours

    def discretize_contour(self, contour, interval=1.0):
        """将完整轮廓路径上连续离散化，确保点间距严格为interval（误差小于1e-3mm）"""
        if not contour:
            return []
            
        # 将轮廓点转换为numpy数组以便计算
        points = np.array(contour)
        
        # 如果轮廓只有一个点，直接返回
        if len(points) < 2:
            return [tuple(points[0])]
        
        # 检查轮廓是否闭合
        is_closed = np.allclose(points[0], points[-1], atol=1e-3)
        
        # 如果轮廓是闭合的，我们需要确保离散点也形成闭合轮廓
        if is_closed:
            # 对于闭合轮廓，我们需要确保离散点也闭合
            # 首先计算所有相邻点之间的距离，包括从最后一个点回到第一个点的距离
            segment_vectors = np.diff(points, axis=0)
            if len(segment_vectors) > 0:
                # 添加从最后一个点回到第一个点的向量
                closing_vector = points[0] - points[-1]
                segment_vectors = np.vstack([segment_vectors, closing_vector])
            
            # 计算所有线段的距离
            segment_distances = np.linalg.norm(segment_vectors, axis=1)
            
            # 计算累积距离
            cumulative_distances = np.zeros(len(segment_distances) + 1)
            cumulative_distances[1:] = np.cumsum(segment_distances)
            total_length = cumulative_distances[-1]
            
            if total_length < 1e-6:
                return [tuple(points[0])]
            
            discrete_points = []
            
            # 从起点开始，每隔interval生成一个点
            target_distance = 0.0
            current_segment = 0
            
            # 生成均匀分布的点
            while target_distance <= total_length - interval / 2:
                # 找到目标距离所在的线段
                while current_segment < len(cumulative_distances) - 1:
                    if cumulative_distances[current_segment] <= target_distance <= cumulative_distances[current_segment + 1]:
                        break
                    current_segment += 1
                
                # 确保当前线段有效
                if current_segment >= len(cumulative_distances) - 1:
                    break
                
                # 获取当前线段的起点和终点
                p0 = points[current_segment % len(points)]
                p1 = points[(current_segment + 1) % len(points)]
                
                # 计算在线段内的位置比例
                segment_start = cumulative_distances[current_segment]
                segment_end = cumulative_distances[current_segment + 1]
                segment_length = segment_end - segment_start
                
                if segment_length < 1e-6:
                    # 跳过非常短的线段
                    target_distance += interval
                    continue
                
                t = (target_distance - segment_start) / segment_length
                t = np.clip(t, 0.0, 1.0)  # 确保t在0-1范围内
                
                # 计算当前点坐标
                current_point = p0 + t * (p1 - p0)
                
                # 检查是否是重复点（避免因为浮点数误差导致的重复）
                if not discrete_points or not np.allclose(discrete_points[-1], current_point, atol=1e-3):
                    discrete_points.append(tuple(current_point))
                
                # 移动到下一个目标距离
                target_distance += interval
            
            # 确保闭合轮廓的离散点也闭合（最后一个点和第一个点距离很近）
            if discrete_points and len(discrete_points) > 1:
                first_point = np.array(discrete_points[0])
                last_point = np.array(discrete_points[-1])
                distance_to_start = np.linalg.norm(last_point - first_point)
                
                # 如果最后一个点离起点太远，添加一个接近起点的点
                if distance_to_start > interval / 2:
                    # 计算从最后一个点到起点的方向向量
                    direction = first_point - last_point
                    direction_norm = direction / np.linalg.norm(direction)
                    
                    # 计算接近起点的点
                    closing_point = last_point + direction_norm * (interval - distance_to_start)
                    
                    # 确保闭合点接近起点但不重复
                    if not np.allclose(closing_point, first_point, atol=1e-3):
                        discrete_points.append(tuple(closing_point))
        else:
            # 处理开放轮廓
            # 计算所有相邻点之间的距离
            segment_vectors = np.diff(points, axis=0)
            segment_distances = np.linalg.norm(segment_vectors, axis=1)
            
            # 计算累积距离
            cumulative_distances = np.zeros(len(segment_distances) + 1)
            cumulative_distances[1:] = np.cumsum(segment_distances)
            total_length = cumulative_distances[-1]
            
            if total_length < 1e-6:
                return [tuple(points[0])]
            
            discrete_points = []
            
            # 从起点开始，每隔interval生成一个点
            target_distance = 0.0
            current_segment = 0
            
            # 生成均匀分布的点
            while target_distance <= total_length:
                # 找到目标距离所在的线段
                while current_segment < len(cumulative_distances) - 1:
                    if cumulative_distances[current_segment] <= target_distance <= cumulative_distances[current_segment + 1]:
                        break
                    current_segment += 1
                
                # 确保当前线段有效
                if current_segment >= len(cumulative_distances) - 1:
                    break
                
                # 获取当前线段的起点和终点
                p0 = points[current_segment]
                p1 = points[current_segment + 1]
                
                # 计算在线段内的位置比例
                segment_start = cumulative_distances[current_segment]
                segment_end = cumulative_distances[current_segment + 1]
                segment_length = segment_end - segment_start
                
                if segment_length < 1e-6:
                    # 跳过非常短的线段
                    target_distance += interval
                    continue
                
                t = (target_distance - segment_start) / segment_length
                t = np.clip(t, 0.0, 1.0)  # 确保t在0-1范围内
                
                # 计算当前点坐标
                current_point = p0 + t * (p1 - p0)
                discrete_points.append(tuple(current_point))
                
                # 移动到下一个目标距离
                target_distance += interval
        
        return discrete_points
    
    def _filter_grid_points_by_contour_distance(self, grid_points, contour_points):
        """过滤掉与轮廓点距离小于阈值的网格点
        
        Args:
            grid_points: 网格点列表 [(x1,y1), (x2,y2), ...]
            contour_points: 轮廓点列表 [(x1,y1), (x2,y2), ...]
            
        Returns:
            过滤后的网格点列表
        """
        if not grid_points or not contour_points:
            return grid_points
        
        # 转换为numpy数组以便向量化计算
        grid_array = np.array(grid_points)
        contour_array = np.array(contour_points)
        
        filtered_points = []
        
        # 对每个网格点，计算到所有轮廓点的最小距离
        for grid_point in grid_array:
            # 计算到所有轮廓点的距离
            distances = np.linalg.norm(contour_array - grid_point, axis=1)
            min_distance = np.min(distances)
            
            # 如果最小距离大于阈值，保留该点
            if min_distance >= self.filter_distance:
                filtered_points.append(tuple(grid_point))
        
        print(f"  网格点过滤前: {len(grid_points)} 个")
        print(f"  网格点过滤后: {len(filtered_points)} 个")
        print(f"  过滤掉 {len(grid_points) - len(filtered_points)} 个近距离点")
        
        return filtered_points

    def visualize_slice(self, z_height):
        """可视化指定z高度的切片结果"""
        # 创建输出目录
        output_dir = "stl_slices"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 获取切片线段
        slices = self.slice_mesh(z_height)
        
        if not slices:
            print(f"在z={z_height}高度没有切片结果")
            return
        
        # 重建轮廓
        contours = self.reconstruct_contours(slices)
        
        if not contours:
            print(f"在z={z_height}高度无法重建轮廓")
            return
        
        # 准备绘图
        fig, ax = plt.subplots()
        
        # 绘制原始切片线段
        for segment in slices:
            x = [segment[0][0], segment[1][0]]
            y = [segment[0][1], segment[1][1]]
            ax.plot(x, y, 'b-', linewidth=1, alpha=0.5)
        
        # 绘制重建的轮廓和离散化点
        all_discrete_points = []
        all_contour_points = []
        
        for i, contour in enumerate(contours):
            # 绘制重建的轮廓
            if len(contour) > 1:
                contour_x = [point[0] for point in contour]
                contour_y = [point[1] for point in contour]
                ax.plot(contour_x, contour_y, 'r-', linewidth=0, alpha=0.8)
            
            # 离散化轮廓
            discrete_points = self.discretize_contour(contour, self.discretize_interval)
            all_discrete_points.extend(discrete_points)
            all_contour_points.extend(discrete_points)
            
            # 绘制离散化点
            if discrete_points:
                discrete_x = [point[0] for point in discrete_points]
                discrete_y = [point[1] for point in discrete_points]
                ax.scatter(discrete_x, discrete_y, color='red', s=1, label=f'')
        
        # 生成并绘制网格点阵（蓝色）
        if contours:
            # 查找轮廓层次关系
            hierarchy = self._find_contour_hierarchy(contours)
            
            for outer_idx, inner_idxs in hierarchy['hierarchy']:
                outer_contour = contours[outer_idx]
                inner_contours_list = [contours[idx] for idx in inner_idxs]
                
                # 生成网格点阵
                grid_points = self._generate_grid_points(outer_contour, inner_contours_list, self.discretize_interval)
                
                # 过滤掉与轮廓点距离小于20mm的网格点
                filtered_grid_points = self._filter_grid_points_by_contour_distance(grid_points, all_contour_points)
                
                # 绘制网格点阵
                if filtered_grid_points:
                    grid_x = [point[0] for point in filtered_grid_points]
                    grid_y = [point[1] for point in filtered_grid_points]
                    ax.scatter(grid_x, grid_y, color='red', s=1, label='')
                    
                    # 将过滤后的网格点添加到所有离散点中
                    all_discrete_points.extend(filtered_grid_points)
        
        ax.set_aspect('equal')
        ax.set_title(f'')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.grid(True)
        ax.legend()
        
        # 保存图像
        image_file = os.path.join(output_dir, f"slice_z_{z_height:.2f}mm_filtered.png")
        plt.savefig(image_file, dpi=300, bbox_inches='tight')
        
        # 保存离散化点
        discrete_file = os.path.join(output_dir, f"slice_z_{z_height:.2f}mm_discrete_filtered.txt")
        with open(discrete_file, 'w') as f:
            f.write(f"Slice at z={z_height:.2f}mm\n")
            f.write(f"Scale factor: {self.scale_factor}x\n")
            f.write(f"Discretize interval: {self.discretize_interval}mm\n")
            f.write(f"Filter distance: {self.filter_distance}mm\n")
            f.write(f"Number of discrete points: {len(all_discrete_points)}\n")
            f.write("\nDiscrete points:\n")
            for i, point in enumerate(all_discrete_points):
                f.write(f"{i+1}: ({point[0]:.4f}, {point[1]:.4f})\n")
        
        plt.close()
        print(f"过滤后的切片结果已保存到 {image_file}")
        print(f"过滤后的离散化点已保存到 {discrete_file}")

    def generate_slices(self, layer_height=1.0):
        """生成一系列切片"""
        # 获取模型的Z轴范围
        min_z = np.min(self.mesh.points[:, 2])
        max_z = np.max(self.mesh.points[:, 2])
        
        print(f"模型Z轴范围: {min_z:.2f}mm - {max_z:.2f}mm")
        
        # 生成切片
        z = min_z
        while z <= max_z:
            self.visualize_slice(z)
            z += layer_height

    def _get_contour_area(self, contour):
        """计算闭合轮廓的面积（用于判断方向：正为逆时针/外轮廓，负为顺时针/内轮廓）
        Args:
            contour: 闭合轮廓点列表 [(x1,y1), (x2,y2), ...]
        Returns:
            轮廓面积（mm²）
        """
        if len(contour) < 3:
            return 0.0
        
        area = 0.0
        n = len(contour)
        
        for i in range(n):
            x1, y1 = contour[i]
            x2, y2 = contour[(i+1) % n]  # 循环到第一个点
            area += (x1 * y2 - x2 * y1)
        
        return area / 2.0
    
    def _is_point_in_polygon(self, point, polygon):
        """判断点是否在多边形内部（射线法）
        Args:
            point: (x, y) 点坐标
            polygon: 多边形点列表 [(x1,y1), (x2,y2), ...]
        Returns:
            True: 点在多边形内部；False: 点在多边形外部
        """
        x, y = point
        inside = False
        
        n = len(polygon)
        for i in range(n):
            j = (i + 1) % n
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            
            # 检查点是否在边上（简化处理）
            if abs((yj - yi) * x - (xj - xi) * y + xj * yi - yj * xi) < 1e-6:
                return False  # 点在边上，不算内部
            
            # 射线法判断
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
        
        return inside
    
    def _find_contour_hierarchy(self, contours):
        """查找轮廓的嵌套层次关系
        Args:
            contours: 轮廓列表 [[(x1,y1), ...], [(x2,y2), ...], ...]
        Returns:
            层次字典 {'outer_contours': [], 'inner_contours': [], 'hierarchy': [(外轮廓索引, [内轮廓索引列表]), ...]}
        """
        outer_contours = []
        inner_contours = []
        hierarchy = []
        
        # 首先按面积绝对值排序（从大到小）
        sorted_contours = sorted(
            enumerate(contours),
            key=lambda x: abs(self._get_contour_area(x[1])),
            reverse=True
        )
        
        # 最大的轮廓肯定是外轮廓
        if sorted_contours:
            outer_idx, outer_contour = sorted_contours[0]
            outer_contours.append(outer_idx)
            hierarchy.append((outer_idx, []))
            
            # 检查其他轮廓是否是内轮廓
            for contour_idx, contour in sorted_contours[1:]:
                # 如果轮廓面积与外轮廓符号相反，说明是内轮廓
                outer_area = self._get_contour_area(outer_contour)
                current_area = self._get_contour_area(contour)
                
                if (outer_area > 0 and current_area < 0) or (outer_area < 0 and current_area > 0):
                    # 进一步确认是否在外轮廓内部
                    if self._is_point_in_polygon(contour[0], outer_contour):
                        inner_contours.append(contour_idx)
                        hierarchy[0][1].append(contour_idx)
        
        return {
            'outer_contours': outer_contours,
            'inner_contours': inner_contours,
            'hierarchy': hierarchy
        }
    
    def _generate_grid_points(self, outer_contour, inner_contours, interval=30.0):
        """在外轮廓内部、内轮廓外部生成网格点阵
        Args:
            outer_contour: 外轮廓点列表
            inner_contours: 内轮廓点列表
            interval: 点阵间隔（mm）
        Returns:
            网格点阵列表 [(x1,y1), (x2,y2), ...]
        """
        grid_points = []
        
        if not outer_contour:
            return grid_points
        
        # 计算外轮廓的边界框
        x_coords = [p[0] for p in outer_contour]
        y_coords = [p[1] for p in outer_contour]
        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)
        
        # 生成网格点
        x = x_min
        while x <= x_max:
            y = y_min
            while y <= y_max:
                point = (x, y)
                
                # 检查点是否在外轮廓内部
                if not self._is_point_in_polygon(point, outer_contour):
                    y += interval
                    continue
                
                # 检查点是否在任何内轮廓内部（如果是则排除）
                inside_any_inner = False
                for inner_contour in inner_contours:
                    if self._is_point_in_polygon(point, inner_contour):
                        inside_any_inner = True
                        break
                
                if not inside_any_inner:
                    grid_points.append(point)
                
                y += interval
            
            x += interval
        
        return grid_points
    
    def generate_gcode_file(self, z_height, output_file="output_filtered.gcode"):
        """生成过滤后的G代码文件
        Args:
            z_height: 切片高度
            output_file: 输出文件名
        """
        print(f"正在为z={z_height}mm生成过滤后的G代码...")
        gcode = self.generate_gcode(z_height)
        
        if gcode:
            with open(output_file, 'w') as f:
                f.write(gcode)
            print(f"过滤后的G代码已保存到 {output_file}")
        else:
            print(f"无法在z={z_height}高度生成G代码")
            
    def generate_gcode(self, z_height):
        """生成指定z高度的过滤后的G代码
        Args:
            z_height: 切片高度
        Returns:
            G代码字符串
        """
        # 获取切片线段
        slices = self.slice_mesh(z_height)
        print(f"  切片线段数量: {len(slices)}")
        
        if not slices:
            print(f"  在z={z_height}高度没有切片结果")
            return ""
        
        # 重建轮廓
        contours = self.reconstruct_contours(slices)
        print(f"  重建轮廓数量: {len(contours)}")
        
        if not contours:
            print(f"  在z={z_height}高度无法重建轮廓")
            return ""
        
        # 准备G代码
        gcode_lines = []
        gcode_lines.append("G90 ; 使用绝对坐标")
        gcode_lines.append("G17 ; XY平面")
        gcode_lines.append("G21 ; 使用毫米单位")
        gcode_lines.append(f"G0 Z{z_height + 5.0:.2f} ; 快速移动到安全高度")
        
        # 处理轮廓和过滤后的网格点
        all_points = []
        all_contour_points = []
        
        for i, contour in enumerate(contours):
            # 离散化轮廓
            discrete_points = self.discretize_contour(contour, self.discretize_interval)
            print(f"  轮廓 {i+1} 离散点数量: {len(discrete_points)}")
            all_points.extend(discrete_points)
            all_contour_points.extend(discrete_points)
        
        # 生成网格点并过滤
        hierarchy = self._find_contour_hierarchy(contours)
        print(f"  轮廓层次结构: {hierarchy}")
        
        for outer_idx, inner_idxs in hierarchy['hierarchy']:
            outer_contour = contours[outer_idx]
            inner_contours_list = [contours[idx] for idx in inner_idxs]
            
            # 生成网格点阵
            grid_points = self._generate_grid_points(outer_contour, inner_contours_list, self.discretize_interval)
            
            # 过滤掉与轮廓点距离小于20mm的网格点
            filtered_grid_points = self._filter_grid_points_by_contour_distance(grid_points, all_contour_points)
            print(f"  过滤后的网格点阵数量: {len(filtered_grid_points)}")
            
            # 添加过滤后的网格点
            all_points.extend(filtered_grid_points)
        
        print(f"  过滤后的总G代码点数量: {len(all_points)}")
        
        # 如果有离散点，生成G代码
        if all_points:
            # 移动到第一个点（快速移动）
            first_point = all_points[0]
            gcode_lines.append(f"G0 X{first_point[0]:.2f} Y{first_point[1]:.2f} ; 快速移动到第一个点")
            gcode_lines.append(f"G0 Z{z_height:.2f} ; 下落到打印高度")
            
            # 开始打印
            gcode_lines.append("M104 S200 ; 设置挤出温度")
            gcode_lines.append("M109 S200 ; 等待挤出温度达到")
            gcode_lines.append("M107 ; 关闭风扇")
            gcode_lines.append("G1 F1000 ; 设置打印速度")
            
            # 打印所有点（作为连续线条）
            for point in all_points[1:]:
                gcode_lines.append(f"G1 X{point[0]:.2f} Y{point[1]:.2f}")
            
            # 结束打印
            gcode_lines.append("M104 S0 ; 关闭挤出加热")
            # 先计算抬升高度，确保变量正确替换
            lift_z = z_height + 10.0
            gcode_lines.append(f"G0 Z{lift_z:.2f} ; 抬升喷嘴")
        
        gcode = "\n".join(gcode_lines)
        print(f"  生成的G代码行数: {len(gcode_lines)}")
        return gcode

if __name__ == "__main__":
    import os
    import numpy as np
    stl_file = os.path.expanduser("~/Desktop/001.stl")
    
    if not os.path.exists(stl_file):
        print(f"文件 {stl_file} 不存在，请确保STL文件在桌面上")
        exit(1)
    
    slicer = STLSlicer(stl_file)
    
    # 打印模型的z轴范围
    min_z = np.min(slicer.mesh.points[:, 2])
    max_z = np.max(slicer.mesh.points[:, 2])
    print(f"模型Z轴范围: {min_z:.2f}mm - {max_z:.2f}mm")
    
    # 选择一个合适的z高度进行测试
    test_z = 4.65  # 选择中间位置
    print(f"选择测试z高度: {test_z:.2f}mm")
    
    # 测试可视化功能（过滤版本）
    slicer.visualize_slice(test_z)
    print("过滤版本可视化测试完成！")
    
    # 测试G代码生成功能（过滤版本）
    slicer.generate_gcode_file(test_z, "test_output_filtered.gcode")
    print("过滤版本G代码生成测试完成！")
    
    print("所有测试完成！")