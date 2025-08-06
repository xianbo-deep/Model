from torch import nn
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter
from torch import optim
import torch
from tqdm import tqdm
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from model import *
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = resnet50(10).to(device)

train_data = datasets.CIFAR10(root='./data', train=True, transform=train_transform, download=False)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True,num_workers=4,pin_memory=True)
test_data = datasets.CIFAR10(root='./data', train=False, transform=test_transform, download=False)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False,num_workers=4,pin_memory=True)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


criterion = nn.CrossEntropyLoss()


epochs = 20

writer = SummaryWriter('./logs/resnet50')

best_acc = 0

def evaluate(data_loader):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total += loss.item()
            correct += calculate_accuracy(outputs, labels)
    return total / len(data_loader), correct / len(data_loader)



def calculate_accuracy(outputs,labels):
    _,preds = torch.max(outputs, dim=1)
    correct = (preds == labels).sum().item()
    return correct / len(labels)

global_steps = 0
for epoch in range(epochs):
    model.train()
    train_process = tqdm(train_loader,desc=f'Epoch {epoch + 1}/{epochs}',leave=True)
    for batch_idx, (inputs, labels) in enumerate(train_process):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        accuracy = calculate_accuracy(outputs, labels)

        writer.add_scalar('train/loss', loss.item(), global_steps)
        writer.add_scalar('train/accuracy', accuracy, global_steps)

        train_process.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{accuracy:.4f}'
        })

        global_steps += 1

    eval_loss,eval_accuracy = evaluate(test_loader)
    writer.add_scalar('test/loss', eval_loss, epoch)
    writer.add_scalar('test/accuracy', eval_accuracy, epoch)

    if eval_accuracy > best_acc:
        best_acc = eval_accuracy
        torch.save(model.state_dict(), './model/resnet50.pth')
    scheduler.step()


writer.close()
