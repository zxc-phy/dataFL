import numpy as np
import torch
import copy
from torch.utils.data import DataLoader
from sim.utils.utils import AverageMeter, accuracy
import random
import math

### 定义节点
class SensorNodes():
    '''Sensor nodes in the network        
    '''
    def __init__(self, num_nodes, square_size = 100):
      self.num_nodes = num_nodes  # 节点数量
      self.square_size = square_size  # 正方形边长
      self.nodes = self.generate_nodes()  # 生成节点
      self.adj_list = self.generate_adj_list()  # 生成邻接列表
      self.node_pull_history = {i : [] for i in self.num_nodes}   # 存储节点的历史数据

    def generate_nodes(self):
        """ 生成节点，每个节点在正方形区域内有一个随机坐标。"""
        return {i: (random.uniform(0, self.square_size), random.uniform(0, self.square_size)) for i in range(self.num_nodes)}

    def generate_adj_list(self):
        """生成邻接列表表示的图。随机决定节点间是否有连接，并存储距离。"""
        adj_list = {i: [] for i in range(self.num_nodes)}
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                # 假设每对节点有 20% 的几率连接
                if random.random() < 0.2:
                    distance = self.calculate_distance(self.nodes[i], self.nodes[j])
                    adj_list[i].append((j, distance))
                    adj_list[j].append((i, distance))
        return adj_list 
    
    def calculate_distance(self, coord1, coord2):
        """计算两点之间的欧氏距离。"""
        return math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)
   
    def display_graph(self):
        """ 打印图的邻接列表和节点坐标。"""
        print("Node Coordinates:")
        for node, coords in self.nodes.items():
            print(f"Node {node}: {coords}")
        print("\nAdjacency List:")
        for node, neighbors in self.adj_list.items():
            print(f"Node {node}: {neighbors}")


    def get_total_datasizes(self, net_dataidx_map) :
        '''Get the total datasize of each node
        '''
        self.SN_datasizes = {i : len(net_dataidx_map[i]) for i in range(self.num_nodes)}
        return self.SN_datasizes 
    
    def collect_datasizes(self, selected_nodes, episode):
        r'''Collect the datasizes of selected nodes
          每个节点的数据是动态积累的
        Args:
            selected_clients(list(int)): the SNs to collect data from at episode E




            
            episode(int): Current episode
        Returns:
          data_ratio(dict):{selected SN id(int) : datasize ratio}, e.g., {0:0.5,3:0.6}
        '''
        

        self.SN_collect_datasizes = {i : self.SN_datasizes[i] for i in selected_nodes}
        return self.SN_collect_datasizes
