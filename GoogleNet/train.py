from torchvision import datasets,models
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.optim import lr_scheduler
from torch import optim
from torchvision import transforms
from model import *

device = torch.device("cuda:0"if torch.cuda.is_available() else "cpu")

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.1),  # 添加垂直翻转
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomGrayscale(p=0.1),  # 随机灰度化
    transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.1),  # 高斯模糊
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.2))  # 随机擦除
])

val_test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
model = GoogLeNet(num_classes=102,aux_logits=True,init_weights=True).to(device)
# model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1).to(device)

writer = SummaryWriter('./logs')

train_data = datasets.Flowers102(root='./data',split="train",transform=train_transform,download=True)
test_data = datasets.Flowers102(root='./data',split="test",transform=val_test_transform,download=True)
val_data = datasets.Flowers102(root='./data',split="val",transform=val_test_transform,download=True)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True,num_workers=4,pin_memory=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False,num_workers=4,pin_memory=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False,num_workers=4,pin_memory=True)


epochs = 100

criterion = nn.CrossEntropyLoss().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4,weight_decay=0.01)  # 增大学习率

scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

global_steps = 0


def calculate_accuracy(output, labels):
    preds = torch.argmax(output, dim=1)
    correct = (preds==labels).sum().item()

    return correct / len(labels)


def eval(data_loader):
    model.eval()
    total_loss = 0.
    total_accuracy = 0.
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            correct = calculate_accuracy(outputs, labels)
            total_accuracy += correct


    return total_loss / len(data_loader), total_accuracy / len(data_loader)

best_acc = 0
for epoch in range(epochs):
    model.train()
    train_process = tqdm(train_loader,desc = f"Epoch {epoch + 1}/{epochs}",leave = True)
    for batch_idx, (inputs,labels) in enumerate(train_process):
        inputs,labels = inputs.to(device),labels.to(device)

        # 梯度清零
        optimizer.zero_grad()
        outputs,aux1,aux2 = model(inputs)
        # outputs = model(inputs)
        # 计算损失
        loss_main = criterion(outputs, labels)
        loss_aux1 = criterion(aux1, labels)
        loss_aux2 = criterion(aux2, labels)
        total_loss = loss_main + 0.3 * loss_aux1 + 0.3 * loss_aux2
        # total_loss = criterion(outputs, labels)
        # 计算正确率
        accuracy = calculate_accuracy(outputs, labels)

        # 反向传播
        total_loss.backward()

        # 更新参数
        optimizer.step()

        # 写入tensorboard

        writer.add_scalar('train/loss', total_loss.item(), global_steps)
        writer.add_scalar('train/accuracy', accuracy, global_steps)

        train_process.set_postfix({
            'loss': f'{total_loss.item():.4f}',
            'acc': f'{accuracy:.4f}'
        })

        global_steps += 1

    # 调整学习率
    scheduler.step()

    # 测试
    val_loss, val_acc = eval(val_loader)
    writer.add_scalar('test/loss', val_loss, epoch)
    writer.add_scalar('test/accuracy', val_acc, epoch)
    if val_acc > best_acc:
        torch.save(model.state_dict(), './model/GoogleNet_best.pth')
        best_acc = val_acc


model.load_state_dict(torch.load('./model/GoogleNet_best.pth'))
test_loss, test_acc = eval(test_loader)
print(f'Test Accuracy: {test_acc:.2%}')
writer.close()