import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = f.max_pool2d(f.relu(self.conv1(x)), (2, 2))

        # If the size is a square you can only specify a single number
        x = f.max_pool2d(f.relu(self.conv2(x)), 2)

        x = x.view(-1, self.num_flat_features(x))
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        y = self.fc3(x)

        return y

    @staticmethod
    def num_flat_features(x):
        # all dimensions except the batch dimension
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)

# params = list(net.parameters())
# print(len(params))
# print(params[0].size())  # weight of conv1
#
# input = torch.randn(1, 1, 32, 32)
# print(input.size())
# output = net(input)
# target = torch.randn(10).view(1, -1)
#
# criterion = nn.MSELoss()
# loss = criterion(output, target)
# print(loss.grad_fn)
# print(loss.grad_fn.next_functions[0][0])  # Linear
# print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU
#
# net.zero_grad()
# print("conv1 bias.grad before backward")
# print(net.conv1.bias.grad)
#
# loss.backward()
#
# print("conv1 bias.grad after backward")
# print(net.conv1.bias.grad)

