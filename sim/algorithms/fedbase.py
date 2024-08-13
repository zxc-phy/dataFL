import numpy as np
import torch
import copy
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from sim.utils.utils import AverageMeter, accuracy
import torchvision.transforms as transforms
import random
from sim.utils.record_utils import logconfig, add_log, record_exp_result2

from operator import attrgetter
from heapq import nlargest, nsmallest
import sim.algorithms.arm as arm
import sim.utils.bandit_functions as bandit
from sim.utils.options import args_parser
from scipy.stats import entropy

args = args_parser()
eval_batch_size = 32
num_class_dict = { 'mnist': 10, 'fashionmnist': 10, 'cifar10': 10, 'cifar100': 100, 'cinic10': 10, 'test': 4, 'svhn': 10, 'har': 6, 'animal': 10, 'ham':7}

# t: episode
def CalcUCB(myarm, z, t, M, Z):
    beta = 0.5
    m = myarm.model
    pred = m.predict(np.array([[z]]))
    cb = np.sqrt(2*np.log(t**2*Z*M))
    # 缩小方差
    cb = cb/1000
    ucb = pred[0] + cb*np.sqrt(pred[1])
    myarm.ucb = ucb
    myarm.grad = np.mean(myarm.grad_hist)
    myarm.qual = beta * myarm.ucb + (1 - beta) * myarm.grad

def size_divergence(dataset, label_proportions):
    num_classes = num_class_dict[args.d]
    labels = np.array([label for _, label in dataset])
    # print('labels: {}'.format(labels))
    p = np.bincount(labels, minlength=num_classes)
    total_labels = len(labels)
    scaling = 10000
    p = p / total_labels
    # q = np.ones(num_classes) / num_classes
    q = label_proportions
    KL_div = entropy(p, q)
    l2_dist = np.linalg.norm(p - q)
    alpha = 1.0
    data_quality = alpha * l2_dist / total_labels
    return data_quality * scaling

def get_model_gradients(model):
    """
    Get all the gradients of the model and output them as a 1D tensor.

    Parameters:
      model (torch.nn.Module): the model to get the gradients of

    Returns:
      torch.Tensor: a 1D tensor containing all the gradients
    """
    gradients = []
    for param in model.parameters():
        if param.grad is not None:
            gradients.append(param.grad.view(-1))
    return torch.cat(gradients)

def Evaluate(dataset, model, device, label_proportions):
    alpha = 0
    beta = 0.9
    eval_batch_size = 16
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    if len(dataset) >= 512:
        sub_indices = list(range(512))
        eval_subset = Subset(dataset, sub_indices)
        data_loader = DataLoader(eval_subset, batch_size=eval_batch_size, shuffle=True)
        model.train()
        batch_grad_list = []
        num_batches = len(data_loader)
        for batch, (X,y) in enumerate(data_loader):
            X, y = X.to(device), y.to(device) 
            # optimizer.zero_grad()
            # if args.d == 'har':
            #     X = X.unsqueeze(1).float()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            gradients = get_model_gradients(model).cpu()
            batch_grad_list.append(gradients)
            optimizer.zero_grad()
        batch_grad = torch.stack(batch_grad_list) 
        grad_norm = torch.mean(torch.norm(batch_grad, dim=1) ** 2)
        var = torch.var(batch_grad, dim=0)
        # print('var: {}'.format(var))
        variance = torch.sum(var).item()
        grad_quality = alpha * grad_norm + (1 - alpha) * variance
        grad_quality *= 0.01
    else:
        grad_quality = 10
    data_quality = size_divergence(dataset, label_proportions)
    # print('data_quality: {}, grad_quality: {}'.format(data_quality, grad_quality))
    return beta * data_quality + (1-beta) * grad_quality, grad_quality

class CustomDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, target = self.dataset[idx]
        if self.transform:
            data = self.transform(data)
        return data, target

# Custom Gaussian noise function
def add_gaussian_noise(tensor, mean=0.0, std=1.0):
    noise = torch.randn(tensor.size()) * std + mean
    noisy_tensor = tensor + noise
    return torch.clamp(noisy_tensor, 0., 1.)

class AddGaussianNoise:
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise

# Other examples of time series transformations
class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, tensor):
        if np.random.rand() < self.p:
            return tensor.flip(-1)  
        return tensor

class RandomCrop:
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, tensor):
        start = np.random.randint(0, tensor.size(-1) - self.crop_size + 1)
        return tensor[:, :, start:start+self.crop_size]

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, tensor):
        for t in self.transforms:
            tensor = t(tensor)
        return tensor

###### CLIENT ######
class FedClient():
    def __init__(self):
        pass
    
    def setup_criterion(self, criterion):
        self.criterion = criterion

    def setup_train_dataset(self, dataset):
        self.train_feddataset = dataset
    
    def setup_test_dataset(self, dataset):
        self.test_dataset = dataset

    def setup_optim_kit(self, optim_kit):
        self.optim_kit = optim_kit
    
    #client.local_update_step(model=copy.deepcopy(server.global_model), dataset=client.train_feddataset.get_dataset(c_id), num_steps=args.K, device=device, clip=args.clip)
    def local_update_step(self, round, c_id, std, model, num_steps, device, label_proportions, **kwargs):
        dataset=self.train_feddataset.get_dataset(c_id)

        if args.d != 'har':
            additional_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),  
                transforms.RandomCrop(32, padding=4),  
                transforms.Lambda(lambda x: add_gaussian_noise(x, mean=0.0, std=std))  # add gausian noise
            ])
        else:
            additional_transform = Compose([
                RandomHorizontalFlip(p=0.5),
                AddGaussianNoise(mean=0.0, std=std),  
                RandomCrop(crop_size=500) 
            ])

        custom_dataset = CustomDataset(dataset, transform=additional_transform)
        # 
        if round == 0:
            data_quality, grad_quality =  Evaluate(custom_dataset, model, device, label_proportions)
        else:
            data_quality, grad_quality = 0, 10
        data_loader = DataLoader(custom_dataset, batch_size=self.optim_kit.batch_size, shuffle=True)
        optimizer = self.optim_kit.optim(model.parameters(), **self.optim_kit.settings)

        prev_model = copy.deepcopy(model)
        model.train()
        step_count = 0
        while(True):
            for batch_idx, (input, target) in enumerate(data_loader):
                input = input.to(device)
                target = target.to(device)
                # print(input.size(), input.shape)
                output = model(input)
                loss = self.criterion(output, target)
                # print('loss: {}, step: {}, batch: {}'.format(loss.item(), step_count, batch_idx))
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
            step_count += 1
            if (step_count >= num_steps):
                break

        with torch.no_grad():
            curr_vec = torch.nn.utils.parameters_to_vector(model.parameters())
            prev_vec = torch.nn.utils.parameters_to_vector(prev_model.parameters())
            delta_vec = curr_vec - prev_vec
            if torch.equal(curr_vec, prev_vec):
                print('No update {}'.format(c_id))
            assert step_count == num_steps            
            # add log
            local_log = {}
            return delta_vec, local_log, data_quality, grad_quality

    def local_update_epoch(self, client_model,data, epoch, batchsize):
        pass

    def evaluate_dataset(self, model, dataset, device):
        '''Evaluate on the given dataset'''
        data_loader = DataLoader(dataset, batch_size=eval_batch_size, shuffle=False)
        
        model.eval()
        with torch.no_grad():
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()

            for input, target in data_loader:
                input = input.to(device)
                target = target.to(device)
                output = model(input)
                loss = self.criterion(output, target)
                acc1, acc5 = accuracy(output, target, topk=[1,5])
                # print('loss: {}, target: {}'.format(loss.item(), target.size(0)))
                losses.update(loss.item(), target.size(0))
                top1.update(acc1.item(), target.size(0))
                top5.update(acc5.item(), target.size(0))

            return losses, top1, top5,

###### SERVER ######
class FedServer():
    def __init__(self):
        super(FedServer, self).__init__()
    
    def setup_model(self, model):
        self.global_model = model
    
    def setup_optim_settings(self, **settings):
        self.lr = settings['lr']
    
    # def select_clients(self, num_clients, num_clients_per_round):
    #     '''https://github.com/lx10077/fedavgpy/blob/master/src/trainers/base.py'''
    #     num_clients_per_round = min(num_clients_per_round, num_clients)
    #     return np.random.choice(num_clients, num_clients_per_round, replace=False)
    
    def global_update(self):
        with torch.no_grad():
            param_vec_curr = torch.nn.utils.parameters_to_vector(self.global_model.parameters()) + self.lr * self.delta_avg 
            return param_vec_curr, self.delta_avg
    
    def aggregate_reset(self):
        self.delta_avg = None
        self.weight_sum = torch.tensor(0) 
    
    def aggregate_update(self, local_delta, weight):
        with torch.no_grad():
            if self.delta_avg == None:
                self.delta_avg = torch.zeros_like(local_delta)
            self.delta_avg.add_(weight * local_delta)
            self.weight_sum.add_(weight)
    
    def aggregate_avg(self):
        with torch.no_grad():
            self.delta_avg.div_(self.weight_sum)

    def select_nodes(self, arms, t, M, N, Z, log_flag):
        for arm in arms:
            CalcUCB(arm, arm.z, t, M, Z)
            # If z is 0, then UCB is set to a large value
            if(arm.z == 0):
                arm.ucb = 1e10
        # Get the K arms with the largest UCB value
        top_N_arms = nsmallest(N, arms, key=attrgetter('ucb'))
        list = [arm.arm_id for arm in top_N_arms]
        return list
 
                
                
                
                
                
