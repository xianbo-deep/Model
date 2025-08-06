from torch.utils.data import Dataset,DataLoader
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from model import VGG
import torch.nn as nn
import torch
from tqdm import tqdm
from torch.optim import lr_scheduler

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_data = torchvision.datasets.CIFAR10(root='./data', train=True,download=False,transform=train_transform)
test_data = torchvision.datasets.CIFAR10(root='./data', train=False,download=False,transform=test_transform)

model = VGG(10).to(device)

writer = SummaryWriter("./logs")

train_data = DataLoader(
    train_data,
    batch_size=128,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

test_data = DataLoader(
    test_data,
    batch_size=128,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

optim = torch.optim.Adam(model.parameters(), lr=0.001)


criterion = nn.CrossEntropyLoss().to(device)

scheduler = lr_scheduler.CosineAnnealingLR(optim, T_max=20)

epochs = 20
global_steps = 0
def caculate_accuracy(outputs, labels):
    _ , preds = torch.max(outputs, 1)
    correct = (preds == labels).sum().item()
    return correct / len(labels)


def evaluate(model, data_loader):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluating", leave=False):
            inputs,labels = inputs.to(device),labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            correct += caculate_accuracy(outputs, labels)
    return total_loss / len(data_loader), correct / len(data_loader)

best_acc = 0
for i in range(epochs):
    model.train()
    train_progress = tqdm(train_data, desc=f'Epoch {i + 1}/{epochs}', leave=True)

    for batch_idx, (inputs, labels) in enumerate(train_progress):
        inputs, labels = inputs.to(device), labels.to(device)

        optim.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)


        loss.backward()
        optim.step()

        acc = caculate_accuracy(outputs, labels)
        writer.add_scalar('train/loss', loss.item(), global_steps)
        writer.add_scalar('train/acc', acc, global_steps)
        train_progress.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{acc:.4f}'
        })

        global_steps += 1

    eval_loss,accuracy = evaluate(model, test_data)
    writer.add_scalar('test/loss', eval_loss, i)
    writer.add_scalar('test/acc', accuracy, i)
    if accuracy > best_acc:
        best_acc = accuracy
        torch.save(model.state_dict(), f'./model/VGG.pth')



writer.close()