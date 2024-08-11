from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100, SVHN
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import Dataset, random_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch

def build_dataset(dataset_name='mnist', dataset_dir = '../datasets/'):
    if dataset_name == 'mnist':
        train_dataset, test_dataset = dataset_mnist(dataset_dir)
    elif dataset_name == 'fashionmnist':
        train_dataset, test_dataset = dataset_fashionmnist(dataset_dir)
    elif dataset_name == 'cifar10':
        train_dataset, test_dataset = dataset_cifar10(dataset_dir)
    elif dataset_name == 'cifar100':
        train_dataset, test_dataset = dataset_cifar100(dataset_dir)
    elif dataset_name == 'cinic10':
        train_dataset, test_dataset = dataset_cinic10(dataset_dir)
    elif dataset_name == 'svhn':
        train_dataset, test_dataset = dataset_svhn(dataset_dir)
    elif dataset_name == 'har':
        train_dataset, test_dataset = dataset_har(dataset_dir)
    elif dataset_name == 'animal':
        train_dataset, test_dataset = dataset_animal(dataset_dir)
    elif dataset_name == 'ham':
        train_dataset, test_dataset = dataset_ham(dataset_dir)
    return train_dataset, test_dataset

def dataset_ham(data_path):
    transform = transforms.Compose([
    transforms.Resize((32,32)),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.backends.cudnn.deterministic = True
    # 加载数据集
    full_dataset = datasets.ImageFolder(data_path + 'base_dir/train_dir', transform=transform)

    # 计算训练集和测试集的大小
    total_size = len(full_dataset)
    train_size = int(total_size * 0.8)  # 假设使用80%的数据作为训练集
    test_size = total_size - train_size
    # 划分数据集
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    return train_dataset, test_dataset

def dataset_animal(data_path):
    transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
            ])
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.backends.cudnn.deterministic = True
    full_dataset = datasets.ImageFolder(data_path + 'raw-img', transform=transform)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    return train_dataset, test_dataset

class UCIHAR(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32).unsqueeze(0), torch.tensor(self.labels[idx], dtype=torch.long)

def dataset_har(data_path):
    X_train = pd.read_csv(data_path + 'UCI HAR Dataset/train/X_train.txt', delim_whitespace=True, header=None).values
    y_train = pd.read_csv(data_path + 'UCI HAR Dataset/train/y_train.txt', delim_whitespace=True, header=None).values.ravel()

    # 加载测试集
    X_test = pd.read_csv(data_path + 'UCI HAR Dataset/test/X_test.txt', delim_whitespace=True, header=None).values
    y_test = pd.read_csv(data_path + 'UCI HAR Dataset/test/y_test.txt', delim_whitespace=True, header=None).values.ravel()

    y_train = y_train - 1
    y_test = y_test - 1
    # 标准化数据
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # print(y_train, y_test)
    train_dataset = UCIHAR(X_train, y_train)
    test_dataset = UCIHAR(X_test, y_test)
    return train_dataset, test_dataset

def dataset_svhn(data_path):
    transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
            ]) # mean: 0.13066235184669495, std:0.30810782313346863
    train_dataset = SVHN(root=data_path, split='train', download=True, transform=transform)
    test_dataset = SVHN(root=data_path, split='test', download=True, transform=transform)
    return train_dataset, test_dataset

def dataset_mnist(data_path):
    '''
    https://github.com/AshwinRJ/Federated-Learning-PyTorch/blob/master/src/utils.py
    '''
    transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
            ]) # mean: 0.13066235184669495, std:0.30810782313346863
    train_dataset = MNIST(root=data_path, train=True, download=True, transform=transform)
    test_dataset = MNIST(root=data_path, train=False, download=True, transform=transform)

    return train_dataset, test_dataset


def dataset_fashionmnist(data_path):
    '''
    Replace the mean and std with the data generated manually
    https://github.com/Divyansh03/FedExP/blob/main/util_data.py
    '''
    transform_train = transforms.Compose([
                        transforms.ToTensor(),
                        # transforms.Normalize((0.2860,), (0.3530,))
                        ])
    
    transform_test = transforms.Compose([
                        transforms.ToTensor(),
                        # transforms.Normalize((0.2860,), (0.3530,)),
                        ])

    train_dataset = FashionMNIST(root=data_path, train=True, download=True, transform=transform_train)
    test_dataset = FashionMNIST(root=data_path, train=False, download=True, transform=transform_test)

    return train_dataset, test_dataset


def dataset_cifar10(data_path):
    '''
    https://github.com/JYWa/FedNova/blob/master/util_v4.py
    https://github.com/FedML-AI/FedML/blob/master/python/fedml/data/cifar10/data_loader.py
    '''
    mean = [0.49139968, 0.48215827, 0.44653124]
    std = [0.24703233, 0.24348505, 0.26158768]

    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean, std),
    ])
    
    train_dataset = CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
    test_dataset = CIFAR10(root=data_path, train=False, download=True, transform=transform_test)

    return train_dataset, test_dataset


def dataset_cifar100(data_path):
    '''
    https://github.com/FedML-AI/FedML/blob/master/python/fedml/data/cifar100/data_loader.py
    '''
    mean = [0.5071, 0.4865, 0.4409]
    std = [0.2673, 0.2564, 0.2762]

    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(15),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std)
    ])
    transform_test = transforms.Compose(
        [transforms.ToTensor(),
        # transforms.Normalize(mean, std)
        ])

    train_dataset = CIFAR100(root=data_path, train=True, download=True, transform=transform_train)
    test_dataset = CIFAR100(root=data_path, train=False, download=True, transform=transform_test)
    
    return train_dataset, test_dataset

def dataset_cinic10(data_path):
    '''
    https://github.com/BayesWatch/cinic-10
    https://github.com/Divyansh03/FedExP/blob/main/util_data.py
    '''
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]
    # train_dataset = datasets.ImageFolder('{}/{}'.format(data_path, '/CINIC-10/train/'), transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=cinic_mean,std=cinic_std)]))
    # test_dataset = datasets.ImageFolder('{}/{}'.format(data_path, '/CINIC-10/test/'), transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=cinic_mean,std=cinic_std)]))
    train_dataset = datasets.ImageFolder('{}/{}'.format(data_path, '/CINIC-10/train/'), transform=transforms.Compose([transforms.ToTensor()]))
    test_dataset = datasets.ImageFolder('{}/{}'.format(data_path, '/CINIC-10/test/'), transform=transforms.Compose([transforms.ToTensor()]))
    return train_dataset, test_dataset


if __name__ == '__main__':
    for i in range(0, 3):
        # to judge if the sample sequence is the same at different times
        train_dataset, test_dataset = dataset_mnist('../datasets/')
        print(train_dataset.targets[:30])
   
    
    
