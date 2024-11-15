import torch
from torch import nn

class CNNAQI(nn.Module):
    def __init__(self, num_classes=6):
        super(CNNAQI, self).__init__()
        # 将 in_channels 修改为 1 以适应灰度图像
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        # lgl
        # for debug
        # self.print_input_count = 0
        
        # 修改 fc1 的输入尺寸为 64*13*13 以适应 64x64 的图像输入
        # self.fc1 = nn.Linear(1600, 384)
        self.fc1 = nn.Linear(64*13*13, 384)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(384, 192)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(192, num_classes)

    def forward(self, x):
        # lgl
        # print input size
        # x is a torch Tensor
        # if self.print_input_count < 1:
        #     print("CNNAQI input size: ", x.size())
        #     if x.size() == torch.Size([32, 3, 32, 32]):
        #         import pdb; pdb.set_trace()
        #     self.print_input_count += 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    from pytorch_model_summary import summary
    from model_utils import count_parameters
    
    # 实例化模型
    model = CNNAQI(num_classes=6)
    
    # 计算并打印参数数量
    count_parameters(model)
    
    # 打印模型结构，使用灰度图像 (1, 1, 64, 64) 作为输入
    print(summary(model, torch.zeros((1, 1, 64, 64)), show_input=True))
