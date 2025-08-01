from torch.utils.data import Dataset,DataLoader
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from model import Lenet
import torch.nn as nn
import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_data = torchvision.datasets.MNIST(root='./data', train=True,download=True,transform=transforms)
test_data = torchvision.datasets.MNIST(root='./data', train=False,download=True,transform=transforms)

model = Lenet().to(device)

writer = SummaryWriter("./logs")

train_data = DataLoader(
    train_data,
    batch_size=512,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

test_data = DataLoader(
    test_data,
    batch_size=512,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

optim = torch.optim.SGD(
    model.parameters(),
    lr = 0.001,
    momentum = 0.9,
    weight_decay = 0.001,
)


criterion = nn.CrossEntropyLoss().to(device)

scheduler = StepLR(optim, step_size=10, gamma=0.1)

epochs = 25
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


for i in range(epochs):
    model.train()
    train_progress = tqdm(train_data, desc=f'Epoch {i + 1}/{epochs}', leave=True)

    for batch_idx, (inputs, labels) in enumerate(train_data):
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

torch.save(model.state_dict(), "./model/lenet.pth")

writer.close()