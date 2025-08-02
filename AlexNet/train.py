from tqdm import tqdm
from torchvision import datasets
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from model import *


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(227),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=512, shuffle=True, num_workers=4,pin_memory=True)

test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=512, shuffle=False,num_workers=4,pin_memory=True)

model = AlexNet().to(device)

criterion = nn.CrossEntropyLoss()

writer = SummaryWriter('./logs')

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9,weight_decay=0.001)

scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)


global_steps = 0
epochs = 50


def caculate_accuracy(output, label):
    output = torch.argmax(output, dim=1)
    correct = (output == label).sum().item()
    return correct / len(label)

def eval():
    model.eval()
    total_loss = 0
    total_acc = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader,desc="Evaluating",leave = False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total_acc += caculate_accuracy(outputs, labels)
    return total_loss / len(test_loader), total_acc / len(test_loader)


for i in range(epochs):
    model.train()
    train_tqdm = tqdm(train_loader,desc=f'{i + 1}/{epochs}',leave=True)

    for batch_idx,(inputs,labels) in enumerate(train_tqdm):
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        # 计算损失
        loss = criterion(outputs, labels)
        acc = caculate_accuracy(outputs, labels)
        writer.add_scalar('train/loss',loss.item(),global_steps)
        writer.add_scalar('train/acc',acc,global_steps)

        train_tqdm.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{acc:.4f}'
        })

        # 反向传播
        loss.backward()


        # 更新参数
        optimizer.step()

        # 清零梯度
        optimizer.zero_grad(set_to_none=True)

        global_steps += 1

    loss, acc = eval()
    writer.add_scalar('test/loss',loss,i)
    writer.add_scalar('test/acc',acc,i)

    scheduler.step()

torch.save(model.state_dict(),'./model/AlexNet.pth')

writer.close()