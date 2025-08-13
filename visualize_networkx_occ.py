import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import datetime
import cv2
def visualize_networkx_as_occ(G, resolution=0.5, x_range=(-100, 100), y_range=(-100, 100)):
    """
    将networkX图对象或坐标元组显示为类似占据栅格地图(OCC)的图像
    
    参数:
        G: networkX图对象 或 (coords, labels)元组
        resolution: 栅格分辨率(米)
        x_range: X轴范围(min, max)
        y_range: Y轴范围(min, max)
    """
    # 处理元组输入情况
    if isinstance(G, tuple):
        coords, _ = G  # 假设输入是(coords, labels)元组
        coords = np.array(coords)
        # 创建临时图对象
        G = nx.Graph()
        for i, coord in enumerate(coords):
            G.add_node(i, pos=coord)
    
    # 获取所有节点位置
    pos = nx.get_node_attributes(G, 'pos')
    
    # 如果没有位置属性，使用默认布局
    if not pos:
        pos = nx.spring_layout(G)
    
    # 提取坐标
    coords = np.array(list(pos.values()))
    print(len(coords))
    #print(coords[0:10,:])
    
    # 计算栅格地图尺寸
    x_min, x_max = x_range
    y_min, y_max = y_range
    grid_width = int((x_max - x_min) / resolution)
    grid_height = int((y_max - y_min) / resolution)
    
    # 初始化栅格地图(默认全0表示可通行)
    occ_grid = np.zeros((grid_height, grid_width), dtype=np.uint8)
    
    # 计算每个节点的栅格坐标
    grid_x = np.floor((coords[:, 0] - x_min) / resolution).astype(int)
    grid_y = np.floor((coords[:, 1] - y_min) / resolution).astype(int)
    
    # 过滤超出边界的点
    valid_mask = (grid_x >= 0) & (grid_x < grid_width) & (grid_y >= 0) & (grid_y < grid_height)
    grid_x = grid_x[valid_mask]
    grid_y = grid_y[valid_mask]
    
    # 设置栅格值(1表示障碍物)
    #print(grid_x[0:10])
    #print(grid_y[0:10])
    occ_grid[grid_y, grid_x] = 1
    
    # 创建图形
    plt.figure(figsize=(50, 50))
    
    # 使用OCC风格的颜色方案
    cmap = plt.cm.colors.ListedColormap(['green', 'red'])  # 绿色:可通行, 红色:障碍物
    
    # 显示栅格地图
    plt.imshow(occ_grid, cmap=cmap, interpolation='none', 
              extent=[x_min, x_max, y_min, y_max])
    
  
    
    
    
    # 显示图形
    #plt.show()

    filename = 'out/networkx_OCC_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.png'
    plt.savefig(filename)
    plt.close()


#输入点为为车辆本地OCC坐标(左下角为原点，X轴向前指向车头，Y轴向上)
# startpts,endpts 为车辆本地OCC坐标
def visualize_save_occ_as_vehlococcplt(occvehloc, startpts,endpts,filename_startwith):
        
        occ3 = occvehloc
        plt.imshow(occ3, cmap='gray',origin = 'lower') #坐标系原点在左下角，Y轴向上

        plt.plot(startpts[0], startpts[1], marker='o', color='green')  # 绘制起点
        plt.plot(endpts[0], endpts[1],  markersize=5, marker='x', color='green')  # 绘制终点

        filename = filename_startwith + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.jpg'
        plt.savefig(filename)
        plt.close()


#输入点为车辆本地坐标系，0,0 为车中心，X轴向前指向车头，Y轴向上。需要被转换为车辆本地OCC坐标左下角为原点，X轴向前指向车头，Y轴向上
#occ3 为车辆本地OCC坐标左下角为原点，X轴向前指向车头，Y轴向上
def visualize_save_occ_as_vehlocplt(occ3, vehicle,originpts,filename_startwith,resolution=0.5, x_range=(-100, 100), y_range=(-100, 100)):

    
    plt.imshow(occ3, cmap='gray',origin = 'lower') #坐标系原点在左下角，Y轴向上
   
    plt.title(f"车本地坐标系原点转换为occ网格坐标系后，原点在:{x_range[1]/resolution,y_range[1]/resolution},X轴向右,Y轴向上,\n注意图像进行了翻转确保是为occ坐标系 \n OCC坐标与图像坐标一一对应，但是Y的方向不一样",fontproperties='SimHei')  # 设置标题

    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")


    x_min, x_max = x_range
    y_min, y_max = y_range

    grid_x = np.floor((originpts[0] - x_min) / resolution).astype(int)
    grid_y = np.floor((originpts[1] - y_min) / resolution).astype(int)
    origin= [grid_x,grid_y] 


    bounding_box = vehicle.bounding_box
    bbox_extent = bounding_box.extent
    corners = [
                (-bbox_extent.x, -bbox_extent.x), # 左下角
                (bbox_extent.x, -bbox_extent.x),  # 右下角
                (bbox_extent.x, bbox_extent.x),   # 右上角
                (-bbox_extent.x, bbox_extent.x)   # 左上角
            ]
    # 分离 x 和 y 坐标以供绘图使用
    x_coords, y_coords = zip(*corners)

    # 绘制矩形
    # 使用 `+ [x_coords[0]], + [y_coords[0]]` 来闭合矩形
    plt.plot(list(x_coords) + origin[0], list(y_coords) +  origin[1], marker='o')
    plt.arrow(x=origin[0], y=origin[1], dx = 50,dy = 0, color='yellow', alpha=1)
    plt.arrow(x=origin[0], y=origin[1], dx = 0,dy = 50, color='yellow', alpha=1)


    filename = filename_startwith + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.jpg'
    plt.savefig(filename)
    plt.close()

#输入点为为车辆本地OCC坐标(左下角为原点，X轴向前指向车头，Y轴向上)
# startpts,endpts ,path为车辆本地OCC坐标
def visualize_save_occpath_as_vehlococcplt(occvehloc,path,startpts,endpts,filename_startwith):
        
        occ3 = occvehloc
        plt.imshow(occ3, cmap='gray',origin = 'lower') #坐标系原点在左下角，Y轴向上

        plt.plot(startpts[0], startpts[1], marker='o', color='green')  # 绘制起点
        plt.plot(endpts[0], endpts[1],  markersize=5, marker='x', color='green')  # 绘制终点
        
        #print("path:",path  )
        path_x = path[:,0]
        path_y = path[:,1]
        plt.plot(path_x, path_y, marker='o', color='yellow',markersize=1)  # 绘制起点


        filename = filename_startwith + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.jpg'
        plt.savefig(filename)
        plt.close()
   