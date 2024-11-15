import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import re
import matplotlib.cm as cm
from math import sqrt
# Read the CSV file and extract the selected_node column
file_path = './save/FedAvg_M15_N3_E100_K5_R2_cnnaqi_aqi_0.7_Zmax4_sgd0.03,0.0001_b32_seed1234.csv'
data = pd.read_csv(file_path)
selected_nodes = data['selected_node'].apply(lambda x: list(map(int, re.findall(r'\d+', x))))
print(selected_nodes.head())

# Initialization parameters
grid_size = 20  # Grid Size
base_station_distance = 3  # Minimum distance between base stations
# 初始化图像
plt.figure(figsize=(8, 8), facecolor='white')
plt.xlim(0, grid_size)
plt.ylim(0, grid_size)
plt.axis('off')

def color_map(colors, col_ind):
    new_colors = colors.copy()
    for i, j in enumerate(col_ind):
        new_colors[j] = colors[i]
    return new_colors
    
N = 15          # Number of PoIs
times = np.zeros(N)
M = len(selected_nodes[0])  # Number of UAVs
POIs = np.random.randint(2, grid_size-2, (N, 2))  
# colors = plt.cm.viridis(np.linspace(0, 1, M))
colors = ['#2959A8', '#EE2C2C', '#FFA500']
all_base_stations = np.array([[5, 2], [15, 2], 
                          [10, 18], [2, 10], [18, 10]])  
base_stations = np.array([[5, 2], [15, 2], 
                          [10, 18]])  
initial_positions = base_stations
times[0] += 1
# for poi in POIs:
#     plt.scatter(poi[0], poi[1], color='purple', s=100, zorder=2, label='Poi')  

# for bs in all_base_stations:
#     plt.scatter(bs[0], bs[1], color='black', s=150, marker='^', zorder=3, label='Base Station') 
plt.scatter(POIs[:, 0], POIs[:, 1], color='purple', s=100, zorder=2, label='PoI')  
plt.scatter(all_base_stations[:, 0], all_base_stations[:, 1], color='black', s=150, marker='^', zorder=3, label='ABS')  


# Plotting drone trajectories time-step by time
for time_step in range(0, len(selected_nodes), 2):
    if time_step + 2 >= len(selected_nodes):
        break
    target_positions = np.array([POIs[x] for x in selected_nodes[time_step]])
    
    # Calculate the distance matrix
    cost_matrix = np.linalg.norm(initial_positions[:, None] - target_positions, axis=2)
    
    # Find the best match using the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    #Draw Tracks
    for i, j in zip(row_ind, col_ind):
        # if colors[i] == '#EE2C2C':
        times[selected_nodes[time_step][j]] += 1
        # plt.plot([initial_positions[i, 0], target_positions[j, 0]],
        #         [initial_positions[i, 1], target_positions[j, 1]], 
        #         color=colors[i], marker='o', linestyle='-', linewidth=1, zorder=1)

        # plt.arrow(initial_positions[i, 0], initial_positions[i, 1],
        #           target_positions[j, 0] - initial_positions[i, 0],
        #           target_positions[j, 1] - initial_positions[i, 1],
        #           shape='full', lw=0, length_includes_head=True, head_width=0.2, color=colors[i])
        plt.quiver(initial_positions[i, 0], initial_positions[i, 1],
                target_positions[j, 0] - initial_positions[i, 0],
                target_positions[j, 1] - initial_positions[i, 1],
                angles='xy', scale_units='xy', scale=1, color=colors[i], headwidth=3, headlength=5, headaxislength=4, zorder=1, width=0.005)
    initial_positions = target_positions
    colors = color_map(colors, col_ind)
    base_stations = []
    cost_matrix = np.linalg.norm(initial_positions[:, None] - all_base_stations, axis=2)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    for i, j in zip(row_ind, col_ind):
        # if colors[i] == '#EE2C2C':
            # plt.plot([initial_positions[i, 0], all_base_stations[j, 0]],
            #         [initial_positions[i, 1], all_base_stations[j, 1]], 
            #         color=colors[i], marker='o', linestyle='-', linewidth=1, zorder=1)
        plt.quiver(initial_positions[i, 0], initial_positions[i, 1],
                all_base_stations[j, 0] - initial_positions[i, 0],
                all_base_stations[j, 1] - initial_positions[i, 1],
                angles='xy', scale_units='xy', scale=1, color=colors[i], headwidth=3, headlength=5, headaxislength=4, zorder=1, width=0.005)
        base_stations.append(all_base_stations[j])
    initial_positions = np.array(base_stations)
    if time_step >= 15:
        break

print(times)
# times = np.sqrt(times) * 5
norm = plt.Normalize(times.min(), times.max())  
cmap = cm.Greys  
# circle_colors = cmap(norm(times))
circle_colors = []
for t in times:
    scaled_value = max(0.1, t / 4)  
    circle_colors.append(cmap(scaled_value))

circle_colors = np.array(circle_colors)
# circle_colors[:, :3] = np.clip(circle_colors[:, :3] * 2, 0, 1)
for i, poi in enumerate(POIs):
    radius = 1.5
    circle = plt.Circle(poi, radius, color=circle_colors[i], alpha=0.6, edgecolor='red', linewidth=1.5)
    plt.gca().add_artist(circle)  
# plt.legend(loc='upper right', fontsize=18)
# # plt.title("UAV Trajectory", fontsize=24)
# plt.xlabel("X(km)", fontsize=18)
# plt.ylabel("Y(km)", fontsize=18)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.grid(linestyle='-.', color='gray')
plt.savefig('./save_figure/map1.jpg', dpi=300)