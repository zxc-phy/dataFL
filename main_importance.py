'''FedAvg
Refs:
[1] https://github.com/chandra2thapa/SplitFed
[2] https://github.com/AshwinRJ/Federated-Learning-PyTorch
[3] https://github.com/lx10077/fedavgpy/
[4] https://github.com/shaoxiongji/federated-learning
[5] https://github.com/liyipeng00/convergence
[2024 06 01]'''
import torch
import time
import copy
import re
import numpy as np
import random 

from sim.algorithms.fedbase import FedClient, FedServer
from sim.algorithms.nodes import SensorNodes
from sim.data.data_utils import FedDataset
from sim.data.datasets import build_dataset
from sim.data.partition import build_partition
from sim.models.build_models import build_model
from sim.utils.record_utils import logconfig, add_log, record_exp_result2
from sim.utils.utils import setup_seed, AverageMeter
from sim.utils.optim_utils import OptimKit, LrUpdater
from sim.utils.options import args_parser
from sim.algorithms.arm import Arm
import sim.utils.bandit_functions as bandit
from torch.distributions.dirichlet import Dirichlet
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset

args = args_parser()
num_class_dict = { 'mnist': 10, 'fashionmnist': 10, 'cifar10': 10, 'cifar100': 100, 'cinic10': 10, 'test': 4, 'svhn': 10, 'har': 6, 'animal': 10, 'ham':7}
# nohup python main_fedavg.py -M 10 -N 5 -m mlp -d mnist -s 1 -R 100 -K 10 --partition exdir --alpha 2 10 --optim sgd --lr 0.05 --lr-decay 0.9 --momentum 0 --batch-size 20 --seed 1234 --log Print &
# nohup python main_fedavg.py -M 10 -N 5 -E 20 -m mlp -d mnist  -R 100 -K 10 -Zmax 5 --partition exdir --alpha 2 10 --optim sgd --lr 0.05 --lr-decay 0.9 --momentum 0 --batch-size 20 --seed 1234 --log Print &

# 会有用吗
torch.set_num_threads(4)
setup_seed(args.seed)
device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")

# 删去clip
def customize_record_name(args):
    '''FedAvg_SNs10_MAs10_E5_K2_R4_mlp_mnist_alpha2,10.0_sgd0.001,1.0,0.0,0.0001_b20_seed1234.csv'''
    if args.partition == 'exdir':
        partition = f'{args.partition}{args.alpha[0]},{args.alpha[1]}'
    elif args.partition == 'iid':
        partition = f'{args.partition}'
    
    record_name = f'Import_M{args.M}_N{args.N}_E{args.E}_K{args.K}_R{args.R}_{args.m}_{args.d}_{args.alpha}_Zmax{args.Zmax}'\
                + f'_{args.optim}{args.lr},{args.weight_decay}_b{args.batch_size}_seed{args.seed}'
    return record_name
record_name = customize_record_name(args)

def dirichlet_split_noniid(train_labels, alpha, n_clients):
    n_classes = train_labels.max() + 1
    label_distribution = Dirichlet(torch.full((n_clients,), alpha)).sample((n_classes,))
    # 1. Get the index of each label
    class_idcs = [torch.nonzero(train_labels == y).flatten()
                  for y in range(n_classes)]
    # 2. According to the distribution, the label is assigned to each client
    client_idcs = [[] for _ in range(n_clients)]

    for c, fracs in zip(class_idcs, label_distribution):
        total_size = len(c)
        splits = (fracs * total_size).int()
        splits[-1] = total_size - splits[:-1].sum()
        idcs = torch.split(c, splits.tolist())
        for i, idx in enumerate(idcs):
            client_idcs[i] += [idcs[i]]

    client_idcs = [torch.cat(idcs) for idcs in client_idcs]
    return client_idcs

# 全部节点在某个episode存下来的数据，且为乱序排放
def get_dataid(net_dataidx_map, SN_datasizes, SN_accumulative_dataratio):
    '''get disordering data id for selected nodes
    Args:
    Returns: episode_dataidx_map: dict, key: node_id, value: data_idx
    '''
    episode_dataidx_map = {}
    for i, idx in net_dataidx_map.items():
        data_num = int(SN_datasizes[i]*SN_accumulative_dataratio.get(i))
        idx_left = idx[np.random.choice(len(idx),data_num,replace=False)]
        episode_dataidx_map[i] = idx_left
    return episode_dataidx_map

def evaluate_data(dataset, model, device):
    eval_batch_size = 16
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.03)
    # accumulated_grad = None
    data_loader = DataLoader(dataset, batch_size=eval_batch_size, shuffle=True)
    prev_model = copy.deepcopy(model)
    model.train()
    num_batches = len(data_loader)
    local_K = 2
    step_count = 0
    while(True):
        for batch, (X,y) in enumerate(data_loader):
            X, y = X.to(device), y.to(device) 
            # optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            # batch_grad = [p.grad.view(-1) for p in model.parameters()]
            # batch_grad = torch.cat(batch_grad)
            # if accumulated_grad is None:
            #     accumulated_grad = batch_grad
            # else:
            #     accumulated_grad += batch_grad
            optimizer.step()
            # accumulated_grad /= num_batches
        step_count += 1
        if step_count >= local_K:
            break
    curr_vec = torch.nn.utils.parameters_to_vector(model.parameters())
    prev_vec = torch.nn.utils.parameters_to_vector(prev_model.parameters())
    delta_vec = curr_vec - prev_vec
    # print('data_quality: {}, grad_quality: {}'.format(data_quality, grad_quality))
    return torch.norm(delta_vec)

# def main():
#     global args, record_name, device

#     logconfig(name = record_name, flag = args.log)
#     add_log('{}'.format(args),flag = args.log)
#     add_log('record_name: {}'.format(record_name),flag = args.log)

#     datatype = args.datatype
#     # SN = SensorNodes(args.M,square_size=100)\
#     # Arm(self, arm_id, zinit, Zmax):zinit是初始值，Zmax是最大值
#     # 初始值取0还是最大值，目前取0？？？
#     SNs = []
#     for i in range(args.M):
#         SNs = SNs + [Arm(i, args.Zmax, args.Zmax)]

#     MA = FedClient()
#     server = FedServer()

#     # noise_list = [random.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]) for _ in range(args.M)]
#     # noise_list = [arm.noise for arm in SNs]
#     noise_list = [0 for _ in range(args.M)]

#     # dataidx_map (dict): { client id (int): data indices (numpy.ndarray) }, e.g., {0: [0,1,4], 1: [2,3,5]}
#     train_dataset, test_dataset = build_dataset(args.d)
#     net_dataidx_map = build_partition(args.d, args.M, args.partition, [args.alpha])
    
#     train_labels = torch.tensor([label for _, label in train_dataset])
#     # print('Train labels:', train_labels)
#     # client_indices = dirichlet_split_noniid(train_labels, args.alpha, args.M)

#     # SN_datasizes (dict): { client id (int): data size (int) }, e.g., {0: 100, 1: 200}
#     SN_datasizes = bandit.get_total_datasizes(net_dataidx_map, args.M)
#     # add_log('SN_datasizes: {}'.format(SN_datasizes), flag = args.log)

#     SN_divergence = bandit.cal_divergence(train_labels, net_dataidx_map)
#     for arm in SNs:
#         arm.div = SN_divergence[arm.arm_id]
#         arm.datatype = datatype[arm.arm_id]
#         arm.noise = noise_list[arm.arm_id]

#     global_model = build_model(model=args.m, dataset=args.d)
#     server.setup_model(global_model.to(device))
#     add_log('{}'.format(global_model), flag = args.log)
    
#     # 轨迹
#     MA_SN_match = []

#     start_time = time.time()

#     # grad_norm_list = np.zeros(args.M)
#     grad_norm_list = torch.zeros(args.M)
#     record_exp_result2(record_name, {'round':0})
#     explore_episode = int(args.M / args.N)
#     for episode in range(0, explore_episode):
#         add_log("------------------episode: {}------------------------------".format(episode),flag=args.log)
#         # print(f"------------------episode: {episode}------------------------------")
#         # construct optim kit
#         MA_optim_kit = OptimKit(optim_name=args.optim, batch_size=args.batch_size,lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
#         MA_optim_kit.setup_lr_updater(LrUpdater.exponential_lr_updater,mul=args.lr_decay)
#         MA.setup_optim_kit(MA_optim_kit) 
#         MA.setup_criterion(torch.nn.CrossEntropyLoss())
#         server.setup_optim_settings(lr = args.global_lr)

#         # slected_nodes = random.sample(range(args.M), args.N)
#         ### 主要的函数
#         # 探索阶段
#         selected_nodes = list(range(episode * args.N, episode * args.N + args.N))
#         # add_log('selected nodes: {} at episode {}'.format(selected_nodes, episode), flag = args.log)
#         # selected_nodes = server.select_nodes(SNs, episode, args.M, args.N, args.Zmax)

#         delta_list = torch.zeros(args.M)

#         # ### 分配MA到SN, 还没写
#         # episode_MA_SN = server.allign_MAs(selected_nodes,MA_SN_match)
#         # MA_SN_match.append(episode_MA_SN)
#         add_log('--------------------selected nodes: {} at episode {}---------------------'.format(selected_nodes, episode), flag = args.log)
#         # print('--------------------selected nodes: {} at episode {}---------------------'.format(selected_nodes, episode))
#         ### 采集数据，arm的设置,,,,,dict 
#         # SN_sccumulative_dataratio: {arm_id: dataratio}
#         SN_accumulative_dataratio = {}
#         for arm in SNs:
#             dataratio = arm.datafunct(arm.z)
#             SN_accumulative_dataratio[arm.arm_id] = dataratio
#             arm.datasize = int(SN_datasizes[arm.arm_id]*dataratio)
#             quality_1 = arm.div/arm.datasize * 10000
#             add_log('arm_id:{}, arm.z:{}, datasize:{}, arm.div:{}, arm.noise: {}, arm.quality: {}, arm.UCB: {}'.format(arm.arm_id,arm.z,arm.datasize, arm.div, arm.noise, quality_1, arm.ucb), flag = args.log)
#             # print('arm_id:{}, arm.z:{}, dataratio:{}'.format(arm.arm_id,arm.z,dataratio))
#             add_log('arm.zhist:{}, arm.yhist:{}'.format(arm.zhist, arm.yhist), flag = args.log)
#         # 观察调试
#         # z_list = [arm.z for arm in SNs]
#         # print('z_list:{} at episode'.format(z_list))
#         # SN_accumulative_dataratio = SN.accumulative_dataratio(selected_nodes,episode)
#         episode_dataidx_map = get_dataid(net_dataidx_map,SN_datasizes,SN_accumulative_dataratio)
#         # datasize_selected = [len(episode_dataidx_map[i]) for i in selected_nodes]
#         # print('datasize_selected:{} at episode'.format(datasize_selected))
#         # train_feddaset.fedsets[i](list): [i上的数据]
#         train_feddataset = FedDataset(train_dataset, episode_dataidx_map)
#         MA.setup_train_dataset(train_feddataset)
#         MA.setup_test_dataset(test_dataset)


#         data_quality = {}
#         for round in range(args.R):
#             print(f"------------------round: {round}------------------------------")
#             server.aggregate_reset()
#             # if round == 0:
#             #     for c_id in selected_nodes:
#             #         quality = MA.
#             for c_id in selected_nodes:
#                 local_delta, local_update_log, quality = MA.local_update_step(round, c_id=c_id,std=noise_list[c_id],model=copy.deepcopy(server.global_model),num_steps=args.K,device=device,clip=args.clip)
#                 delta_list[c_id] += torch.norm(local_delta.cpu())
#                 server.aggregate_update(local_delta,weight=MA.train_feddataset.get_datasetsize(c_id))
#                 if round == 0:
#                     data_quality[c_id] = quality
#             server.aggregate_avg()
#             param_vec_curr = server.global_update()
#             torch.nn.utils.vector_to_parameters(param_vec_curr, server.global_model.parameters())

#             MA.optim_kit.update_lr()
#             add_log('lr={}'.format(MA.optim_kit.settings['lr']), flag=args.log)

#             if (round+1) % max((args.R)//args.eval_num, 1) == 0 or (round+1) > args.R-args.tail_eval_num:
#                 # evaluate on train dataset (selected client)
#                 train_losses, train_top1, train_top5 = AverageMeter(), AverageMeter(), AverageMeter()
#                 for c_id in selected_nodes:
#                     local_losses, local_top1, local_top5 = \
#                     MA.evaluate_dataset(model=server.global_model, dataset=MA.train_feddataset.get_dataset(c_id), device=args.device)
#                     train_losses.update(local_losses.avg, local_losses.count), train_top1.update(local_top1.avg, local_top1.count), train_top5.update(local_top5.avg, local_top5.count)
#                 add_log("Round {}'s server1 train acc: {:6.2f}%, train loss: {:.4f}".format(round+1, train_top1.avg, train_losses.avg), 'green', flag=args.log)
#                 print("Round {}'s server1 train acc: {:6.2f}%, train loss: {:.4f}".format(round+1, train_top1.avg, train_losses.avg))
                
#                 # evaludate on train dataset (random client)
#                 train2_losses, train2_top1, train2_top5 = AverageMeter(), AverageMeter(), AverageMeter()
#                 # rand_eval_clients = server.select_clients(args.M, args.P)
#                 rand_eval_clients = np.random.choice(args.M, args.N, replace = False)
#                 for c_id in rand_eval_clients:
#                     local_losses, local_top1, local_top5 = \
#                     MA.evaluate_dataset(model=server.global_model, dataset=MA.train_feddataset.get_dataset(c_id), device=args.device)
#                     train2_losses.update(local_losses.avg, local_losses.count), train2_top1.update(local_top1.avg, local_top1.count), train2_top5.update(local_top5.avg, local_top5.count)
#                 add_log("Round {}'s server2 train acc: {:6.2f}%, train loss: {:.4f}".format(round+1, train2_top1.avg, train2_losses.avg), 'blue', flag=args.log)
#                 print("Round {}'s server2 train acc: {:6.2f}%, train loss: {:.4f}".format(round+1, train2_top1.avg, train2_losses.avg))
#                 # evaluate on test dataset
#                 test_losses, test_top1, test_top5 = MA.evaluate_dataset(model=server.global_model, dataset=MA.test_dataset, device=args.device)
#                 add_log("Round {}'s server  test  acc: {:6.2f}%, test  loss: {:.4f}".format(round+1, test_top1.avg, test_losses.avg), 'red', flag=args.log)
#                 print("Round {}'s server  test  acc: {:6.2f}%, test  loss: {:.4f}".format(round+1, test_top1.avg, test_losses.avg))
#                 record_exp_result2(record_name, {'round':round+1,
#                                 'train_loss' : train_losses.avg,  'train_top1' : train_top1.avg,  'train_top5' : train_top5.avg, 
#                                 'train2_loss': train2_losses.avg, 'train2_top1': train2_top1.avg, 'train2_top5': train2_top5.avg,
#                                 'test_loss'  : test_losses.avg,   'test_top1'  : test_top1.avg,   'test_top5'  : test_top5.avg })
#         # 每个episode最后做更新
#         for arm in SNs:
#             if arm.arm_id in selected_nodes:
#                 # print(arm.z, data_quality[arm.arm_id])
#                 arm.UpdatePosterior(arm.z,data_quality[arm.arm_id])
#         bandit.UpdateArmZs(SNs, selected_nodes)    
#         grad_norm_list += delta_list

#     for arm in SNs:
#         arm.z = args.Zmax

#     for episode in range(explore_episode, args.E):
#         add_log("------------------episode: {}------------------------------".format(episode),flag=args.log)
#         # construct optim kit
#         MA_optim_kit = OptimKit(optim_name=args.optim,batch_size=args.batch_size,lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
#         MA_optim_kit.setup_lr_updater(LrUpdater.exponential_lr_updater,mul=args.lr_decay)
#         MA.setup_optim_kit(MA_optim_kit) 
#         MA.setup_criterion(torch.nn.CrossEntropyLoss())
#         server.setup_optim_settings(lr=args.global_lr)

#         # slected_nodes = random.sample(range(args.M), args.N)
#         ### 主要的函数
#         selected_nodes = sorted(range(args.M), key=lambda x: grad_norm_list[x], reverse=True)[:args.N]
#         add_log('selected nodes: {} at episode {}'.format(selected_nodes, episode), flag = args.log)
#         # print('--------------------selected nodes: {} at episode {}---------------------'.format(selected_nodes, episode))
#         # ### 分配MA到SN, 还没写
#         # episode_MA_SN = server.allign_MAs(selected_nodes,MA_SN_match)
#         # MA_SN_match.append(episode_MA_SN)
#         # add_log('--------------------selected nodes: {} at episode {}---------------------'.format(selected_nodes, episode), flag = args.log)

#         delta_list = torch.zeros(args.M)
#         ### 采集数据，arm的设置
#         SN_accumulative_dataratio = {}
#         for arm in SNs:
#             dataratio = arm.datafunct(arm.z)
#             SN_accumulative_dataratio[arm.arm_id] = dataratio
#             arm.datasize = int(SN_datasizes[arm.arm_id]*dataratio)
#             quality_1 = arm.div/arm.datasize * 10000
#             add_log('arm_id:{}, arm.z:{}, datasize:{}, arm.div:{}, arm.noise: {}, arm.quality: {}, arm.UCB: {}'.format(arm.arm_id,arm.z,arm.datasize, arm.div, arm.noise, quality_1, arm.ucb), flag = args.log)
#             # print('arm_id:{}, arm.z:{}, dataratio:{}'.format(arm.arm_id,arm.z,dataratio))
#             add_log('arm.zhist:{}, arm.yhist:{}'.format(arm.zhist, arm.yhist), flag = args.log)
#         # SN_accumulative_dataratio = SN.accumulative_dataratio(selected_nodes,episode)
#         # z_list = [arm.z for arm in SNs]
#         # print('z_list:{} at episode'.format(z_list))
#         episode_dataidx_map = get_dataid(net_dataidx_map,SN_datasizes,SN_accumulative_dataratio)
#         # datasize_selected = [len(episode_dataidx_map[i]) for i in selected_nodes]
#         # print('datasize_selected:{} at episode'.format(datasize_selected))

#         train_feddataset = FedDataset(train_dataset, episode_dataidx_map)
#         MA.setup_train_dataset(train_feddataset)
#         MA.setup_test_dataset(test_dataset)


#         data_quality = {}
#         for round in range(args.R):
#             print(f"------------------round: {round}------------------------------")
#             server.aggregate_reset()
#             # if round == 0:
#             #     for c_id in selected_nodes:
#             #         quality = MA.
#             for c_id in selected_nodes:
#                 local_delta, local_update_log, quality = MA.local_update_step(round, c_id=c_id,std=noise_list[c_id],model=copy.deepcopy(server.global_model),num_steps=args.K,device=device,clip=args.clip)
#                 delta_list[c_id] += torch.norm(local_delta.cpu())
#                 server.aggregate_update(local_delta,weight=MA.train_feddataset.get_datasetsize(c_id))
#                 if round == 0:
#                     data_quality[c_id] = quality
#             server.aggregate_avg()
#             param_vec_curr = server.global_update()
#             torch.nn.utils.vector_to_parameters(param_vec_curr, server.global_model.parameters())
            

#             MA.optim_kit.update_lr()
#             add_log('lr={}'.format(MA.optim_kit.settings['lr']), flag=args.log)

#             if (round+1) % max((args.R)//args.eval_num, 1) == 0 or (round+1) > args.R-args.tail_eval_num:
#                 # evaluate on train dataset (selected client)
#                 train_losses, train_top1, train_top5 = AverageMeter(), AverageMeter(), AverageMeter()
#                 for c_id in selected_nodes:
#                     local_losses, local_top1, local_top5 = \
#                     MA.evaluate_dataset(model=server.global_model, dataset=MA.train_feddataset.get_dataset(c_id), device=args.device)
#                     train_losses.update(local_losses.avg, local_losses.count), train_top1.update(local_top1.avg, local_top1.count),  
#                     train_top5.update(local_top5.avg, local_top5.count)
#                 add_log("Round {}'s server1 train acc: {:6.2f}%, train loss: {:.4f}".format(round+1, train_top1.avg, train_losses.avg), 'green', flag=args.log)
#                 print("Round {}'s server1 train acc: {:6.2f}%, train loss: {:.4f}".format(round+1, train_top1.avg, train_losses.avg))
                
#                 # evaludate on train dataset (random client)
#                 train2_losses, train2_top1, train2_top5 = AverageMeter(), AverageMeter(), AverageMeter()
#                 # rand_eval_clients = server.select_clients(args.M, args.P)
#                 rand_eval_clients = np.random.choice(args.M, args.N, replace = False)
#                 for c_id in rand_eval_clients:
#                     local_losses, local_top1, local_top5 = \
#                     MA.evaluate_dataset(model=server.global_model, dataset=MA.train_feddataset.get_dataset(c_id), device=args.device)
#                     train2_losses.update(local_losses.avg, local_losses.count), train2_top1.update(local_top1.avg, local_top1.count), train2_top5.update(local_top5.avg, local_top5.count)
#                 add_log("Round {}'s server2 train acc: {:6.2f}%, train loss: {:.4f}".format(round+1, train2_top1.avg, train2_losses.avg), 'blue', flag=args.log)

#                 print("Round {}'s server2 train acc: {:6.2f}%, train loss: {:.4f}".format(round+1, train2_top1.avg, train2_losses.avg))
#                 # evaluate on test dataset
#                 test_losses, test_top1, test_top5 = MA.evaluate_dataset(model=server.global_model, dataset=MA.test_dataset, device=args.device)
#                 add_log("Round {}'s server  test  acc: {:6.2f}%, test  loss: {:.4f}".format(round+1, test_top1.avg, test_losses.avg), 'red', flag=args.log)
#                 print("Round {}'s server  test  acc: {:6.2f}%, test  loss: {:.4f}".format(round+1, test_top1.avg, test_losses.avg))
#                 record_exp_result2(record_name, {'round':round+1,
#                                 'train_loss' : train_losses.avg,  'train_top1' : train_top1.avg,  'train_top5' : train_top5.avg, 
#                                 'train2_loss': train2_losses.avg, 'train2_top1': train2_top1.avg, 'train2_top5': train2_top5.avg,
#                                 'test_loss'  : test_losses.avg,   'test_top1'  : test_top1.avg,   'test_top5'  : test_top5.avg })
#         # 每个episode最后做更新
#         for arm in SNs:
#             if arm.arm_id in selected_nodes:
#                 arm.UpdatePosterior(arm.z,data_quality[arm.arm_id])
#         bandit.UpdateArmZs(SNs, selected_nodes)    
#         for c_id in selected_nodes:
#             grad_norm_list[c_id] = (grad_norm_list[c_id] * 0.98 + delta_list[c_id]) / (1 + 0.98)

#     if args.save_model == 1:
#       torch.save({'model': torch.nn.utils.parameters_to_vector(server.global_model.parameters())}, './save_model/{}.pt'.format(record_name))
#     end_time = time.time()
#     add_log("TrainingTime: {} sec".format(end_time - start_time), flag=args.log)


def main():
    global args, record_name, device

    logconfig(name = record_name, flag = args.log)
    add_log('{}'.format(args),flag = args.log)
    add_log('record_name: {}'.format(record_name),flag = args.log)

    datatype = args.datatype
    # SN = SensorNodes(args.M,square_size=100)\
    # Arm(self, arm_id, zinit, Zmax):zinit是初始值，Zmax是最大值
    # 初始值取0还是最大值，目前取0？？？
    SNs = []
    for i in range(args.M):
        SNs = SNs + [Arm(i, args.Zmax, args.Zmax)]

    MA = FedClient()
    server = FedServer()

    # noise_list = [random.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]) for _ in range(args.M)]
    # noise_list = [arm.noise for arm in SNs]
    noise_list = args.noise

    # dataidx_map (dict): { client id (int): data indices (numpy.ndarray) }, e.g., {0: [0,1,4], 1: [2,3,5]}
    train_dataset, test_dataset = build_dataset(args.d)
    net_dataidx_map = build_partition(args.d, args.M, args.partition, [args.alpha])
    
    train_labels = torch.tensor([label for _, label in train_dataset])

    labels = [label for _, label in train_dataset]

    # 统计每个类别的数据量
    num_classes = num_class_dict[args.d]
    label_counts = np.bincount(labels, minlength=num_classes)

    # 计算每个类别的数据量在总数据量中的比例
    total_samples = len(labels)
    label_proportions = label_counts / total_samples
    # print('Train labels:', train_labels)
    # client_indices = dirichlet_split_noniid(train_labels, args.alpha, args.M)

    # SN_datasizes (dict): { client id (int): data size (int) }, e.g., {0: 100, 1: 200}
    SN_datasizes = bandit.get_total_datasizes(net_dataidx_map, args.M)
    # add_log('SN_datasizes: {}'.format(SN_datasizes), flag = args.log)

    SN_divergence = bandit.cal_divergence(train_labels, net_dataidx_map, label_proportions)
    for arm in SNs:
        arm.div = SN_divergence[arm.arm_id]
        arm.datatype = datatype[arm.arm_id]
        arm.noise = noise_list[arm.arm_id]

    global_model = build_model(model=args.m, dataset=args.d)
    server.setup_model(global_model.to(device))
    add_log('{}'.format(global_model), flag = args.log)
    
    # 轨迹
    MA_SN_match = []

    start_time = time.time()

    # grad_norm_list = np.zeros(args.M)
    grad_norm_list = torch.zeros(args.M)
    record_exp_result2(record_name, {'round':0})
    explore_episode = int(args.M / args.N)
    for episode in range(0, args.E):
        add_log("------------------episode: {}------------------------------".format(episode),flag=args.log)
        # print(f"------------------episode: {episode}------------------------------")
        # construct optim kit
        MA_optim_kit = OptimKit(optim_name=args.optim, batch_size=args.batch_size,lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        MA_optim_kit.setup_lr_updater(LrUpdater.exponential_lr_updater,mul=args.lr_decay)
        MA.setup_optim_kit(MA_optim_kit) 
        MA.setup_criterion(torch.nn.CrossEntropyLoss())
        server.setup_optim_settings(lr = args.global_lr)

        # slected_nodes = random.sample(range(args.M), args.N)
        ### 主要的函数
        # 探索阶段
        # selected_nodes = list(range(episode * args.N, episode * args.N + args.N))
        # add_log('selected nodes: {} at episode {}'.format(selected_nodes, episode), flag = args.log)
        # selected_nodes = server.select_nodes(SNs, episode, args.M, args.N, args.Zmax)

        
        # ### 分配MA到SN, 还没写
        # episode_MA_SN = server.allign_MAs(selected_nodes,MA_SN_match)
        # MA_SN_match.append(episode_MA_SN)
        
        # print('--------------------selected nodes: {} at episode {}---------------------'.format(selected_nodes, episode))
        ### 采集数据，arm的设置,,,,,dict 
        # SN_sccumulative_dataratio: {arm_id: dataratio}
        SN_accumulative_dataratio = {}
        for arm in SNs:
            dataratio = arm.datafunct(arm.z)
            SN_accumulative_dataratio[arm.arm_id] = dataratio
            arm.datasize = int(SN_datasizes[arm.arm_id]*dataratio)
            quality_1 = arm.div/arm.datasize * 10000
            add_log('arm_id:{}, arm.z:{}, datasize:{}, arm.div:{}, arm.noise: {}, arm.quality: {}, arm.UCB: {}'.format(arm.arm_id,arm.z,arm.datasize, arm.div, arm.noise, quality_1, arm.ucb), flag = args.log)
            # print('arm_id:{}, arm.z:{}, dataratio:{}'.format(arm.arm_id,arm.z,dataratio))
            add_log('arm.zhist:{}, arm.yhist:{}'.format(arm.zhist, arm.yhist), flag = args.log)
        # 观察调试
        # z_list = [arm.z for arm in SNs]
        # print('z_list:{} at episode'.format(z_list))
        # SN_accumulative_dataratio = SN.accumulative_dataratio(selected_nodes,episode)
        episode_dataidx_map = get_dataid(net_dataidx_map,SN_datasizes,SN_accumulative_dataratio)
        # datasize_selected = [len(episode_dataidx_map[i]) for i in selected_nodes]
        # print('datasize_selected:{} at episode'.format(datasize_selected))
        # train_feddaset.fedsets[i](list): [i上的数据]
        train_feddataset = FedDataset(train_dataset, episode_dataidx_map)
        MA.setup_train_dataset(train_feddataset)
        MA.setup_test_dataset(test_dataset)
        if episode % 10 == 0:
            grad_list = []
            for i, arm in enumerate(SNs):
                grad_list.append(evaluate_data(train_feddataset.get_dataset(i), copy.deepcopy(server.global_model), device))
            grad_list = torch.tensor(grad_list)
            pro_list = (grad_list / torch.sum(grad_list)).numpy()
            # print('grad_list:', grad_list)
            # selected_group_nodes = sorted(range(args.M), key=lambda x: grad_list[x], reverse=True)[:(3 * args.N)]
        selected_nodes = np.random.choice(args.M, args.N, replace = False, p = pro_list)
        # selected_nodes = [selected_group_nodes[i] for i in range((episode % 3) * args.N, (episode % 3) * args.N + args.N)]
        # print('selected_group_nodes:', selected_group_nodes)
        # print('selected_nodes:', selected_nodes)
        add_log('--------------------selected nodes: {} at episode {}---------------------'.format(selected_nodes, episode), flag = args.log)
        data_quality = {}
        for round in range(args.R):
            print(f"------------------round: {round}------------------------------")
            server.aggregate_reset()
            # if round == 0:
            #     for c_id in selected_nodes:
            #         quality = MA.
            for c_id in selected_nodes:
                local_delta, local_update_log, quality, quality_1 = MA.local_update_step(round, c_id=c_id,std=noise_list[c_id],model=copy.deepcopy(server.global_model),num_steps=args.K,device=device,label_proportions=label_proportions,clip=args.clip)
                server.aggregate_update(local_delta,weight=MA.train_feddataset.get_datasetsize(c_id))
                if round == 0:
                    data_quality[c_id] = quality
            server.aggregate_avg()
            param_vec_curr, delta_avg = server.global_update()
            torch.nn.utils.vector_to_parameters(param_vec_curr, server.global_model.parameters())

            MA.optim_kit.update_lr()
            add_log('lr={}'.format(MA.optim_kit.settings['lr']), flag=args.log)

            if (round+1) % max((args.R)//args.eval_num, 1) == 0 or (round+1) > args.R-args.tail_eval_num:
                # evaluate on train dataset (selected client)
                train_losses, train_top1, train_top5 = AverageMeter(), AverageMeter(), AverageMeter()
                for c_id in selected_nodes:
                    local_losses, local_top1, local_top5 = \
                    MA.evaluate_dataset(model=server.global_model, dataset=MA.train_feddataset.get_dataset(c_id), device=args.device)
                    train_losses.update(local_losses.avg, local_losses.count), train_top1.update(local_top1.avg, local_top1.count), train_top5.update(local_top5.avg, local_top5.count)
                add_log("Round {}'s server1 train acc: {:6.2f}%, train loss: {:.4f}".format(round+1, train_top1.avg, train_losses.avg), 'green', flag=args.log)
                print("Round {}'s server1 train acc: {:6.2f}%, train loss: {:.4f}".format(round+1, train_top1.avg, train_losses.avg))
                
                # evaludate on train dataset (random client)
                train2_losses, train2_top1, train2_top5 = AverageMeter(), AverageMeter(), AverageMeter()
                # rand_eval_clients = server.select_clients(args.M, args.P)
                rand_eval_clients = np.random.choice(args.M, args.N, replace = False)
                for c_id in rand_eval_clients:
                    local_losses, local_top1, local_top5 = \
                    MA.evaluate_dataset(model=server.global_model, dataset=MA.train_feddataset.get_dataset(c_id), device=args.device)
                    train2_losses.update(local_losses.avg, local_losses.count), train2_top1.update(local_top1.avg, local_top1.count), train2_top5.update(local_top5.avg, local_top5.count)
                add_log("Round {}'s server2 train acc: {:6.2f}%, train loss: {:.4f}".format(round+1, train2_top1.avg, train2_losses.avg), 'blue', flag=args.log)
                print("Round {}'s server2 train acc: {:6.2f}%, train loss: {:.4f}".format(round+1, train2_top1.avg, train2_losses.avg))
                # evaluate on test dataset
                test_losses, test_top1, test_top5 = MA.evaluate_dataset(model=server.global_model, dataset=MA.test_dataset, device=args.device)
                add_log("Round {}'s server  test  acc: {:6.2f}%, test  loss: {:.4f}".format(round+1, test_top1.avg, test_losses.avg), 'red', flag=args.log)
                print("Round {}'s server  test  acc: {:6.2f}%, test  loss: {:.4f}".format(round+1, test_top1.avg, test_losses.avg))
                record_exp_result2(record_name, {'round':round+1,
                                'train_loss' : train_losses.avg,  'train_top1' : train_top1.avg,  'train_top5' : train_top5.avg, 
                                'train2_loss': train2_losses.avg, 'train2_top1': train2_top1.avg, 'train2_top5': train2_top5.avg,
                                'test_loss'  : test_losses.avg,   'test_top1'  : test_top1.avg,   'test_top5'  : test_top5.avg })
        # 每个episode最后做更新
        for arm in SNs:
            if arm.arm_id in selected_nodes:
                # print(arm.z, data_quality[arm.arm_id])
                arm.UpdatePosterior(arm.z,data_quality[arm.arm_id])
        bandit.UpdateArmZs(SNs, selected_nodes)   


    if args.save_model == 1:
      torch.save({'model': torch.nn.utils.parameters_to_vector(server.global_model.parameters())}, './save_model/{}.pt'.format(record_name))
    end_time = time.time()
    add_log("TrainingTime: {} sec".format(end_time - start_time), flag=args.log)

if __name__ == '__main__':
    main()