import numpy as np

from resnet50 import ResNet50
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,WeightedRandomSampler
import torchvision
from torchvision import transforms,datasets

transform = transforms.Compose([
           transforms.RandomHorizontalFlip(), # 随机水平翻转图片
           transforms.ToTensor(), # 转成张量
           transforms.Lambda(lambda t: (t * 2) - 1) # 归一化到[-1,1]
])
image_size = 50
channels = 1
batch_size = 10




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model=ResNet50(num_classes=10).to(device)

# 损失函数
criterion = nn.CrossEntropyLoss()

# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练函数
def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播与优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss

# 验证函数
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(loader.dataset)
    accuracy = correct / total
    return epoch_loss, accuracy

train_data=datasets.ImageFolder(root='./data/Fruit/train',transform=transform)
targets = [sample[1] for sample in train_data]  # 获取每张图片的标签
class_counts = np.bincount(targets)  # 统计每个类别的数量
class_weights = 1. / class_counts    # 类别权重与数量成反比

# 为每个样本赋权重：根据其所属类别的权重分配
sample_weights = [class_weights[target] for target in targets]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=1000, replacement=True)
test_data=datasets.ImageFolder(root='./data/Fruit/test',transform=transform)
train_loader=DataLoader(train_data,batch_size=100,sampler=sampler)
test_loader=DataLoader(test_data,batch_size=5,shuffle=True)
# 训练参数
num_epochs = 10

for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_accuracy = validate(model, test_loader, criterion, device)
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
