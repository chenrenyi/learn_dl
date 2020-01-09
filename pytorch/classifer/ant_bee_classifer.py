from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image
import math

# interactive mode
plt.ion()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def imshow(inp, title=None):
    """展示图片 for tensor"""
    inp = inp.numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def train_model(model, criterion, optimizer, scheduler=None, num_epochs=25):
    """common model train"""
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # when train, backward and step wight
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # change learning rate after one iter on whole data
            if phase == 'train' and scheduler is not None:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # Save best weights
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best weights
    model.load_state_dict(best_model_wts)
    return model


def visualize_model(model, dataloader, num_images=6):
    """展示模型效果"""
    # 保存训练状态，然后置为计算状态
    was_training = model.training
    model.eval()

    image_num = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                image_num += 1
                ax = plt.subplot(math.ceil(num_images / 2), 2, image_num)
                ax.axis('off')
                ax.set_title('predicted {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if image_num == num_images:
                    model.train(mode=was_training)
                    return

        model.train(mode=was_training)


def predicate(model, img_path):
    image = Image.open(img_path)
    image = data_transforms['val'](image)
    output = model(torch.tensor([image]))
    _, preds = torch.max(output)
    return class_names[preds]


# 加载图片
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# data_dir = 'data/hymenoptera_data'
data_dir = '../data/dog_cats'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=0)
               for x in ['train', 'val']}

# 自定义的图片
mine_image_dataset = datasets.ImageFolder('../data/mine', data_transforms['val'])
mine_dataloader = torch.utils.data.DataLoader(mine_image_dataset, batch_size=4, shuffle=False, num_workers=0)

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

# 展示一些图片
inputs, classes = next(iter(dataloaders['train']))
out = torchvision.utils.make_grid(inputs)
# imshow(out, title=[class_names[x] for x in classes])

# 加载预训练模型
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features

# 更改最后一层全连接层为分类器
model_ft.fc = nn.Linear(num_ftrs, len(class_names))
model_ft = model_ft.to(device)

# 定义误差函数，梯度下降算法，学习率改进策略
criterion = nn.CrossEntropyLoss()
optimizer_conv = optim.SGD(model_ft.fc.parameters(), lr=0.001, momentum=0.9)
# optimizer_conv = optim.Adam(model_ft.fc.parameters(), lr=0.1)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

# 进行实际训练
# model_ft.load_state_dict(torch.load('./dog_cats_model.pth'))
model_conv = train_model(model_ft, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=5)
torch.save(model_ft.state_dict(), './dog_cats_model.pth')

# print(next(iter(dataloaders['val'])))

# 展示效果
visualize_model(model_ft, dataloaders['val'])
# visualize_model(model_ft, mine_dataloader, 6)

# 展示特定图片
# print(predicate(model_ft, '../data/mine/none/image001.jpg'))

plt.ioff()
plt.show()
