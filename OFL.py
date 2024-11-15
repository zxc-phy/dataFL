'''OFL: Online Federated Learning
Refs:
[1] https://github.com/chandra2thapa/SplitFed
[2] https://github.com/AshwinRJ/Federated-Learning-PyTorch
[3] https://github.com/lx10077/fedavgpy/
[4] https://github.com/shaoxiongji/federated-learning
[2024 07 07]'''
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

args = args_parser()

# nohup python main_fedavg.py -M 10 -N 5 -m mlp -d mnist -s 1 -R 100 -K 10 --partition exdir --alpha 2 10 --optim sgd --lr 0.05 --lr-decay 0.9 --momentum 0 --batch-size 20 --seed 1234 --log Print &
num_class_dict = { 'mnist': 10, 'fashionmnist': 10, 'cifar10': 10, 'cifar100': 100, 'cinic10': 10, 'test': 4, 'svhn': 10, 'har': 6, 'animal': 10, 'ham':7,'aqi':6}

torch.set_num_threads(4)
setup_seed(args.seed)
device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")


def customize_record_name(args):
    ''' Generate a customized filename for experiment records.
    e.g., FedAvg_SNs10_MAs10_E5_K2_R4_mlp_mnist_alpha2,10.0_sgd0.001,1.0,0.0,0.0001_b20_seed1234.csv'''
    if args.partition == 'exdir':
        partition = f'{args.partition}{args.alpha[0]},{args.alpha[1]}'
    elif args.partition == 'iid':
        partition = f'{args.partition}'
    
    record_name = f'FedAvg_abla_iid_full_M{args.M}_N{args.N}_E{args.E}_K{args.K}_R{args.R}_{args.m}_{args.d}_{args.alpha}_Zmax{args.Zmax}'\
                + f'_{args.optim}{args.lr},{args.weight_decay}_b{args.batch_size}_seed{args.seed}'
    return record_name
record_name = customize_record_name(args)


def dirichlet_split_noniid(train_labels, alpha, n_clients):
    '''
    Divide the data set based on Dirichlet distribution
    Args:
       train_labels (torch.Tensor): Labels of the training dataset.
       alpha (float): Concentration parameter of the Dirichlet distribution.
       n_clients (int): Number of clients to split the data between
       
    Returns:
       list[torch.Tensor]: List of length n_clients, where each element is a tensor
           containing the indices of data samples assigned to that client
    '''
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


    


def get_dataid(net_dataidx_map, SN_datasizes, SN_accumulative_dataratio):
    '''get disordering data id for selected nodes
    Args: 
       net_dataidx_map: dict, key is node ID, value is list of all data indices owned by that node
       SN_datasizes: dict/list, total data size for each node  
       SN_accumulative_dataratio: dict, key is node ID, value is cumulative data ratio for that node
    Returns: episode_dataidx_map: dict, key: node_id, value: data_idx
    '''
    episode_dataidx_map = {}
    for i, idx in net_dataidx_map.items():
        data_num = int(SN_datasizes[i]*SN_accumulative_dataratio.get(i))
        idx_left = idx[np.random.choice(len(idx),data_num,replace=False)]
        episode_dataidx_map[i] = idx_left
    return episode_dataidx_map




def main():
    global args, record_name, device

    logconfig(name = record_name, flag = args.log)
    add_log('{}'.format(args),flag = args.log)
    add_log('record_name: {}'.format(record_name),flag = args.log)
    datatype = args.datatype
  
    SNs = []
    for i in range(args.M):
        SNs = SNs + [Arm(i, args.Zmax, args.Zmax)]

    MA = FedClient()
    server = FedServer()

    noise_list = args.noise


    train_dataset, test_dataset = build_dataset(args.d)
    net_dataidx_map = build_partition(args.d, args.M, args.partition, [args.alpha])
    

    train_labels = torch.tensor([label for _, label in train_dataset])

    labels = [label for _, label in train_dataset]
    # print('labels:', labels)

    # Count the amount of data for each category
    num_classes = num_class_dict[args.d]
    label_counts = np.bincount(labels, minlength=num_classes)

    # Calculate the proportion of each category's data volume in the total data volume
    total_samples = len(labels)
    label_proportions = label_counts / total_samples

    SN_datasizes = bandit.get_total_datasizes(net_dataidx_map, args.M)

    # SN_divergence = {i: 0.0 for i in range(args.M)}
    SN_divergence = bandit.cal_divergence(train_labels, net_dataidx_map, label_proportions)
    for arm in SNs:
        arm.div = SN_divergence[arm.arm_id]
        arm.datatype = datatype[arm.arm_id]
        arm.noise = noise_list[arm.arm_id]


    global_model = build_model(model=args.m, dataset=args.d)
    server.setup_model(global_model.to(device))
    add_log('{}'.format(global_model), flag = args.log)

    start_time = time.time()

    record_exp_result2(record_name, {'round':0})
    if args.M % args.N == 0:
        explore_episode = int(args.M / args.N)
    else:
        explore_episode = int(args.M / args.N) + 1

    explore_nodes = list(range(args.M))

    # warm up
    for episode in range(0, explore_episode):
        add_log("------------------episode: {}------------------------------".format(episode),flag=args.log)
        # construct optim kit
        MA_optim_kit = OptimKit(optim_name=args.optim, batch_size=args.batch_size,lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        MA_optim_kit.setup_lr_updater(LrUpdater.exponential_lr_updater,mul=args.lr_decay)
        MA.setup_optim_kit(MA_optim_kit) 
        MA.setup_criterion(torch.nn.CrossEntropyLoss())
        server.setup_optim_settings(lr = args.global_lr)

        # if episode * args.N + args.N < args.M:
        #     selected_nodes = list(range(episode * args.N, episode * args.N + args.N))
        # else:
        #     selected_nodes = list(range(episode * args.N, args.M)) + list(range(0, (episode * args.N + args.N) % args.M))
        # print('selected_nodes:', selected_nodes)

        # Random exploration
        selected_nodes = random.sample(explore_nodes, args.N)
        for nodes in selected_nodes:
            explore_nodes.remove(nodes)

        print('selected_nodes:', selected_nodes)


        add_log('selected nodes: {} at episode {}'.format(selected_nodes, episode), flag = args.log)
        print('--------------------selected nodes: {} at episode {}---------------------'.format(selected_nodes, episode))
        
        
        # data collection
        # SN_accumulative_dataratio: {arm_id: dataratio}
        SN_accumulative_dataratio = {}
        for arm in SNs:
            dataratio = arm.datafunct(arm.z)
            SN_accumulative_dataratio[arm.arm_id] = dataratio
            arm.datasize = int(SN_datasizes[arm.arm_id]*dataratio)
            quality_1 = arm.div/arm.datasize * 10000
            add_log('arm_id:{}, arm.z:{}, datasize:{}, arm.div:{}, arm.noise: {}, arm.quality: {}, arm.UCB: {}'.format(arm.arm_id,arm.z,arm.datasize, arm.div, arm.noise, quality_1, arm.ucb), flag = args.log)
            
        episode_dataidx_map = get_dataid(net_dataidx_map,SN_datasizes,SN_accumulative_dataratio)

        train_feddataset = FedDataset(train_dataset, episode_dataidx_map)
        MA.setup_train_dataset(train_feddataset)
        MA.setup_test_dataset(test_dataset)


        data_quality = {}
        grad_quality = {}
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
                    grad_quality[c_id] = quality_1
                    add_log('arm_id:{},arm.z: {}, arm.noise: {}, grad_quality: {}'.format(c_id, SNs[c_id].z,SNs[c_id].noise, quality_1), flag = args.log)

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
                selected_nodes = np.array(selected_nodes)
                test_losses, test_top1, test_top5 = MA.evaluate_dataset(model=server.global_model, dataset=MA.test_dataset, device=args.device)
                add_log("Round {}'s server  test  acc: {:6.2f}%, test  loss: {:.4f}".format(round+1, test_top1.avg, test_losses.avg), 'red', flag=args.log)
                print("Round {}'s server  test  acc: {:6.2f}%, test  loss: {:.4f}".format(round+1, test_top1.avg, test_losses.avg))
                record_exp_result2(record_name, {'round':round+1,
                                'train_loss' : train_losses.avg,  'train_top1' : train_top1.avg,  'train_top5' : train_top5.avg, 
                                'train2_loss': train2_losses.avg, 'train2_top1': train2_top1.avg, 'train2_top5': train2_top5.avg,
                                'test_loss'  : test_losses.avg,   'test_top1'  : test_top1.avg,   'test_top5'  : test_top5.avg, 'selected_node': selected_nodes })
        # update for GP model
        for arm in SNs:
            if arm.arm_id in selected_nodes:
                # print(arm.z, data_quality[arm.arm_id])
                arm.UpdatePosterior(arm.z,data_quality[arm.arm_id], grad_quality[arm.arm_id])
        bandit.UpdateArmZs(SNs, selected_nodes)    

    # PoI selection
    for episode in range(explore_episode, args.E):
        add_log("------------------episode: {}------------------------------".format(episode),flag=args.log)
        # construct optim kit
        MA_optim_kit = OptimKit(optim_name=args.optim,batch_size=args.batch_size,lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
        MA_optim_kit.setup_lr_updater(LrUpdater.exponential_lr_updater,mul=args.lr_decay)
        MA.setup_optim_kit(MA_optim_kit) 
        MA.setup_criterion(torch.nn.CrossEntropyLoss())
        server.setup_optim_settings(lr=args.global_lr)

        selected_nodes = server.select_nodes(SNs, episode, args.M, args.N, args.Zmax,args.log)
        add_log('selected nodes: {} at episode {}'.format(selected_nodes, episode), flag = args.log)
        print('--------------------selected nodes: {} at episode {}---------------------'.format(selected_nodes, episode))

        SN_accumulative_dataratio = {}
        for arm in SNs:
            dataratio = arm.datafunct(arm.z)
            SN_accumulative_dataratio[arm.arm_id] = dataratio
            arm.datasize = int(SN_datasizes[arm.arm_id]*dataratio)
            quality_1 = arm.div/arm.datasize * 10000
            add_log('arm_id:{}, arm.z:{}, datasize:{}, arm.div:{}, arm.noise: {}, arm.quality: {}, arm.UCB: {}'.format(arm.arm_id,arm.z,arm.datasize, arm.div, arm.noise, quality_1, arm.ucb), flag = args.log)
        episode_dataidx_map = get_dataid(net_dataidx_map,SN_datasizes,SN_accumulative_dataratio)

        train_feddataset = FedDataset(train_dataset, episode_dataidx_map)
        MA.setup_train_dataset(train_feddataset)
        MA.setup_test_dataset(test_dataset)


        data_quality = {}
        grad_quality = {}
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
                    grad_quality[c_id] = quality_1
                    add_log('arm_id:{},arm.z: {}, arm.noise: {}, grad_quality: {}'.format(c_id, SNs[c_id].z,SNs[c_id].noise, quality_1), flag = args.log)
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
                    train_losses.update(local_losses.avg, local_losses.count), train_top1.update(local_top1.avg, local_top1.count),  
                    train_top5.update(local_top5.avg, local_top5.count)
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
                selected_nodes = np.array(selected_nodes)
                test_losses, test_top1, test_top5 = MA.evaluate_dataset(model=server.global_model, dataset=MA.test_dataset, device=args.device)
                add_log("Round {}'s server  test  acc: {:6.2f}%, test  loss: {:.4f}".format(round+1, test_top1.avg, test_losses.avg), 'red', flag=args.log)
                print("Round {}'s server  test  acc: {:6.2f}%, test  loss: {:.4f}".format(round+1, test_top1.avg, test_losses.avg))
                record_exp_result2(record_name, {'round':round+1,
                                'train_loss' : train_losses.avg,  'train_top1' : train_top1.avg,  'train_top5' : train_top5.avg, 
                                'train2_loss': train2_losses.avg, 'train2_top1': train2_top1.avg, 'train2_top5': train2_top5.avg,
                                'test_loss'  : test_losses.avg,   'test_top1'  : test_top1.avg,   'test_top5'  : test_top5.avg, 'selected_node': selected_nodes })
        # update GP model
        for arm in SNs:
            if arm.arm_id in selected_nodes:
                arm.UpdatePosterior(arm.z,data_quality[arm.arm_id], grad_quality[arm.arm_id])
        bandit.UpdateArmZs(SNs, selected_nodes)    

    if args.save_model == 1:
      torch.save({'model': torch.nn.utils.parameters_to_vector(server.global_model.parameters())}, './save_model/{}.pt'.format(record_name))
    end_time = time.time()
    add_log("TrainingTime: {} sec".format(end_time - start_time), flag=args.log)

if __name__ == '__main__':
    main()