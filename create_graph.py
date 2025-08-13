import networkx as nx
import numpy as np
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
def create_graph_from_occupancy_grid(occ2, resolution=0.5, obstacle_buffer=0.5):



    # 复制输入的occupancy grid以供后续操作
    occ3 = occ2.copy()
    
    # 获取occupancy grid的行数和列数
    rows, cols = occ2.shape
    
    # 创建二维网格图，通常会看到 (0, 0) 在左上角，X轴向右，Y轴向下
    G = nx.grid_2d_graph(rows, cols)
    
    # 定义节点位置，使(0, 0)位于左下角
    pos = {(i, j): (j, rows - i - 1) for i in range(rows) for j in range(cols)}
    
    # 计算障碍物缓冲距离对应的网格单元数
    buffer_cells = int(obstacle_buffer / resolution)
    
    # 使用NumPy找到所有障碍物位置
    obstacle_positions = np.argwhere(occ2 == 255)
    
    # 预生成所有可能的偏移量
    dr = np.arange(-buffer_cells, buffer_cells + 1)
    dc = np.arange(-buffer_cells, buffer_cells + 1)
    dr_grid, dc_grid = np.meshgrid(dr, dc)
    offsets = np.column_stack((dr_grid.ravel(), dc_grid.ravel()))
    
    # 计算所有需要移除的节点
    all_nodes = set()
    for (r, c) in obstacle_positions:
        # 生成所有缓冲节点
        nodes = offsets + [r, c]
        
        # 过滤超出边界的节点
        valid_nodes = nodes[
            (nodes[:, 0] >= 0) & (nodes[:, 0] < rows) & 
            (nodes[:, 1] >= 0) & (nodes[:, 1] < cols)
        ]
        
        # 添加到待移除集合
        all_nodes.update(map(tuple, valid_nodes))
        
        # 更新occ3中的值，表示缓冲区
        occ3[valid_nodes[:, 0], valid_nodes[:, 1]] = 255
    
    # 批量移除节点
    G.remove_nodes_from(all_nodes)
    
    return G, occ3, pos

# 示例使用
if __name__ == "__main__":  # 主程序入口
    np.random.seed(42)
    occ2 = np.random.choice([0, 255], size=(100, 100), p=[0.7, 0.3])

    G, occ3, pos = create_graph_from_occupancy_grid(occ2, resolution=0.5, obstacle_buffer=1)

    # 输出结果验证
    print("Graph Nodes:", G.nodes())
    print("Updated Occupancy Grid:\n", occ3)


    plt.figure(figsize=(8, 8))
    nx.draw(G, pos=pos, with_labels=True, node_color='lightblue', font_size=10, node_size=500)
    plt.title("Grid Graph with Obstacles and Buffer Zones")
    plt.axis('equal')  # 保持比例一致
    plt.show()

    plt.figure()
    occ3_flipped = np.flipud(occ3)  # 上下翻转图像数据
    plt.imshow(occ3_flipped, cmap='gray', origin='lower')  # 坐标系原点在左下角，Y轴向上
    plt.show()