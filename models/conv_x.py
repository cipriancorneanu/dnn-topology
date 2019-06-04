import torch
import torch.nn as nn
import torch.nn.functional as F

    
class Conv_2(nn.Module):
    def __init__(self, num_classes=10, input_size=28):
        super(Conv_2, self).__init__()
        self.feat_size = 12544 if input_size==32 else 12544 if input_size==28 else -1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(self.feat_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x, mask=None):
        x1 = self.conv1(x)
        if mask: x1 = x1*mask[0]
        x2 = F.relu(F.max_pool2d(x1, 1))
        x3 = self.conv2(x2)
        if mask: x3 = x3*mask[1]
        x4 = F.relu(F.max_pool2d(x3, 2))
        x4 = x4.view(-1, self.feat_size)
        x5 = F.relu(self.fc1(x4))
        if mask: x5 = x5*mask[2]
        x6 = F.relu(self.fc2(x5))
        if mask: x6 = x6*mask[3]
        x7 = F.log_softmax(self.fc3(x6), dim=1)
        return x7
        
    def forward_features(self, x):
        x1 = F.relu(F.max_pool2d(self.conv1(x), 1))
        x2 = F.relu(F.max_pool2d(self.conv2(x1), 2))        
        x2 = x2.view(-1, self.feat_size)
        x3 = F.relu(self.fc1(x2))
        x4 = F.relu(self.fc2(x3))
        x5 = F.log_softmax(self.fc3(x4), dim=1)
        return [x2, x3, x4, x5]

    def forward_param_features(self, x):
        x1 = self.conv1(x)
        x2 = F.relu(F.max_pool2d(x1, 1))
        x3 = self.conv2(x2)
        x4 = F.relu(F.max_pool2d(x3, 2))
        x4 = x4.view(-1, self.feat_size)
        x5 = F.relu(self.fc1(x4))
        x6 = F.relu(self.fc2(x5))
        x7 = F.log_softmax(self.fc3(x6), dim=1)
        return [x1, x3, x5, x6, x7]
        
        
class Conv_4(nn.Module):
    def __init__(self, num_classes=10, input_size=28):
        super(Conv_4, self).__init__()
        self.feat_size = 3200 if input_size==32 else 6272 if input_size==28 else -1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(self.feat_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x, mask=None):
        x1 = self.conv1(x)
        if mask: x1 = x1*mask[0]
        x2 = F.relu(F.max_pool2d(x1, 1))
        x3 = self.conv2(x2)
        if mask: x3 = x3*mask[1]
        x4 = F.relu(F.max_pool2d(x3, 2))
        x5 = self.conv3(x4)
        if mask: x5 = x5*mask[2]
        x6 = F.relu(F.max_pool2d(x5, 1))
        x7 = self.conv4(x6)
        if mask: x7 = x7*mask[3]
        x8 = F.relu(F.max_pool2d(x7, 2))
        x8 = x8.view(-1, self.feat_size)
        x9 = F.relu(self.fc1(x8))
        if mask: x9 = x9*mask[4]
        x10 = F.relu(self.fc2(x9))
        if mask: x10 = x10*mask[5]
        x11 = F.log_softmax(self.fc3(x10), dim=1)
        return x11

    def forward_features(self, x):
        x1 = F.relu(F.max_pool2d(self.conv1(x), 1))
        x2 = F.relu(F.max_pool2d(self.conv2(x1), 2))
        x3 = F.relu(F.max_pool2d(self.conv3(x2), 1))
        x4 = F.relu(F.max_pool2d(self.conv4(x3), 2))
        x4 = x4.view(-1, self.feat_size)
        x5 = F.relu(self.fc1(x4))
        x6 = F.relu(self.fc2(x5))
        x7 = F.log_softmax(self.fc3(x6), dim=1)
        return [x2, x4, x5, x6, x7]

    def forward_param_features(self, x):
        x1 = self.conv1(x)
        x2 = F.relu(F.max_pool2d(x1, 1))
        x3 = self.conv2(x2)
        x4 = F.relu(F.max_pool2d(x3, 2))
        x5 = self.conv3(x4)
        x6 = F.relu(F.max_pool2d(x5, 1))
        x7 = self.conv4(x6)
        x8 = F.relu(F.max_pool2d(x7, 2))
        x8 = x8.view(-1, self.feat_size)
        x9 = F.relu(self.fc1(x8))
        x10 = F.relu(self.fc2(x9))
        x11 = F.log_softmax(self.fc3(x10), dim=1)
        return [x1, x3, x5, x7, x9, x10, x11]


class Conv_6(nn.Module):
    def __init__(self, num_classes=10, input_size=28):
        super(Conv_6, self).__init__()
        self.feat_size = 3200 if input_size==32 else 2304 if input_size==28 else -1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(self.feat_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, num_classes)
    '''
    def forward(self, x):
        x1 = F.relu(F.max_pool2d(self.conv1(x), 1))
        x2 = F.relu(F.max_pool2d(self.conv2(x1), 2))
        x3 = F.relu(F.max_pool2d(self.conv3(x2), 1))
        x4 = F.relu(F.max_pool2d(self.conv4(x3), 2))
        x5 = F.relu(F.max_pool2d(self.conv5(x4), 1))
        x6 = F.relu(F.max_pool2d(self.conv6(x5), 2))
        x6 = x6.view(-1, self.feat_size)
        x7 = F.relu(self.fc1(x6))
        x8 = F.relu(self.fc2(x7))
        x9 = F.log_softmax(self.fc3(x8), dim=1)
        return x9
    '''
    def forward(self, x, mask=None):
        x1 = self.conv1(x)
        if mask: x1 = x1*mask[0]
        x2 = F.relu(F.max_pool2d(x1, 1))
        x3 = self.conv2(x2)
        if mask: x3*mask[1]
        x4 = F.relu(F.max_pool2d(x3, 2))
        x5 = self.conv3(x4)
        if mask: x5*mask[2]
        x6 = F.relu(F.max_pool2d(x5, 1))
        x7 = self.conv4(x6)
        if mask: x7 = x7*mask[3]
        x8 = F.relu(F.max_pool2d(x7, 2))
        x9 = self.conv5(x8)
        if mask: x9 = x9*mask[4]
        x10 = F.relu(F.max_pool2d(x9, 1))
        x11 = self.conv6(x10)
        if mask: x11 = x11*mask[5]
        x12 = F.relu(F.max_pool2d(x11, 2))
        x12 = x12.view(-1, self.feat_size)
        x13 = F.relu(self.fc1(x12))
        if mask: x13 = x13*mask[6] 
        x14 = F.relu(self.fc2(x13))
        if mask: x14 = x14*mask[7]
        x15 = F.log_softmax(self.fc3(x14), dim=1)
        return x15
    
    def forward_features(self, x):
        x1 = F.relu(F.max_pool2d(self.conv1(x), 1))
        x2 = F.relu(F.max_pool2d(self.conv2(x1), 2))
        x3 = F.relu(F.max_pool2d(self.conv3(x2), 1))
        x4 = F.relu(F.max_pool2d(self.conv4(x3), 2))
        x5 = F.relu(F.max_pool2d(self.conv5(x4), 1))
        x6 = F.relu(F.max_pool2d(self.conv6(x5), 2))
        x6 = x6.view(-1, self.feat_size)
        x7 = F.relu(self.fc1(x6))
        x8 = F.relu(self.fc2(x7))
        x9 = F.log_softmax(self.fc3(x8), dim=1)
        return [x5, x6, x7, x8, x9]

    def forward_param_features(self, x):
        x1 = self.conv1(x)
        x2 = F.relu(F.max_pool2d(x1, 1))
        x3 = self.conv2(x2)
        x4 = F.relu(F.max_pool2d(x3, 2))
        x5 = self.conv3(x4)
        x6 = F.relu(F.max_pool2d(x5, 1))
        x7 = self.conv4(x6)
        x8 = F.relu(F.max_pool2d(x7, 2))
        x9 = self.conv5(x8)
        x10 = F.relu(F.max_pool2d(x9, 1))
        x11 = self.conv6(x10)
        x12 = F.relu(F.max_pool2d(x11, 2))
        x12 = x12.view(-1, self.feat_size)
        x13 = F.relu(self.fc1(x12))
        x14 = F.relu(self.fc2(x13))
        x15 = F.log_softmax(self.fc3(x14), dim=1)
        return [x1, x3, x5, x7, x9, x11, x13, x14, x15]


def test():
    import numpy as np 
    conv_2 = Conv_2(num_classes=10, input_size=28)
    conv_4 = Conv_4(num_classes=10, input_size=28)
    conv_6 = Conv_6(num_classes=10, input_size=28)

    x = torch.randn(1,1,28,28)

    print('feat size conv_2')
    y = conv_2(x)
    print('----------')
    print(conv_2)
    print(np.sum([np.prod(x.size()) for x in conv_2.forward_features(x)]))
    for i, layer in enumerate(conv_2.forward_features(x)):
        print('layer {} has size {}'.format(i, layer.shape))

        
    print('feat_size conv_4')
    y = conv_4(x)
    print('---------')
    print(conv_4)
    print(np.sum([np.prod(x.size()) for x in conv_4.forward_features(x)]))
    for i, layer in enumerate(conv_4.forward_features(x)):
        print('layer {} has size {}'.format(i, layer.shape))

    print('feat size conv_6')
    y = conv_6(x)
    print('---------')
    print(conv_6)
    print(np.sum([np.prod(x.size()) for x in conv_6.forward_features(x)]))
    for i, layer in enumerate(conv_6.forward_features(x)):
        print('layer {} has size {}'.format(i, layer.shape))

        
'''test()'''
