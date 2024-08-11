import numpy as np
from scipy.stats import entropy

def get_total_datasizes( net_dataidx_map, num_nodes) :
    '''Get the total datasize of each node
    '''
    SN_datasizes = {i : len(net_dataidx_map[i]) for i in range(num_nodes)}
    return SN_datasizes 

# rewards: dict, key is the node id, value is the reward
def UpdateArmZs(allarms,selected_nodes):
    for arm in allarms:
        if arm.arm_id in selected_nodes:
            arm.z = 1
        else:
            arm.z = min(arm.z+1, arm.Zmax)

            # arm.numplays += 1
            # arm.zhist = np.vstack((arm.zhist, arm.z))
            # arm.yhist = np.vstack((arm.yhist, rewards[arm.arm_id]))
        

# 计算任何一个arm和均匀分布之间的距离
def cal_divergence(train_labels, client_idcs, label_proportions):
    n_labels = train_labels.max().item() + 1  # Fixed typo here
    # n_labels = n_labels.item()
    # uniform = np.ones(n_labels) / n_labels
    uniform = label_proportions
    divergence = {}
    for i, idx in client_idcs.items():
        if len(idx) == 0:  # Check if the list is empty
            divergence[i] = float(0.)  # or some other indicative value
            continue

        distribution = np.zeros(n_labels)
        for j in idx:
            label = train_labels[j.item()]  # Ensure j is a scalar tensor
            distribution[label] += 1
        distribution /= len(idx)
        # print(type(distribution))
        # print(type(uniform))
        
        kl_div = entropy(distribution, uniform)
        l2_dist = np.linalg.norm(distribution - uniform)
        divergence[i] = l2_dist
        # print(i, distribution, divergence[i])
    return divergence
