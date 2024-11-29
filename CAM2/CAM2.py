import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import cv2

##########################################################################################################################
train_model = False
image_index = 100

##########################################################################################################################

# 데이터 전처리 및 로드
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = torchvision.datasets.OxfordIIITPet(root='./data', split='trainval', download=True, transform=transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

test_dataset = torchvision.datasets.OxfordIIITPet(root='./data', split='test', download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Pretrained 모델 불러오기 (ResNet18)
model = models.resnet18(pretrained=True)
num_in_features = model.fc.in_features

# 마지막 레이어 수정하기 - 클래스 수는 37개 (Oxford-IIIT Pet 데이터셋)
num_classes = 37
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, num_classes),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(num_classes, num_classes)
)

# Global Average Pooling 추가
class CustomResNet18(nn.Module):
    def __init__(self, base_model, num_classes):
        super(CustomResNet18, self).__init__()
        self.features = nn.Sequential(*list(base_model.children())[:-2])  # 마지막 두 레이어(GAP와 fc 제외)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = CustomResNet18(model, num_classes)

# Loss 및 Optimizer 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# GPU 사용 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 학습 결과 저장 폴더 생성
if not os.path.exists('./CAM2'):
    os.makedirs('./CAM2')


if train_model:
    # 모델 학습 루프
    epochs = 10
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        test_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in tqdm(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = train_loss / len(data_loader)
        train_accuracy = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        # print(f"Epoch [{epoch + 1}/{epochs}] | Train Loss: {train_loss}, Train Accuracy: {train_accuracy}%")s

        # 테스트 데이터셋에서 성능 테스트
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        test_accuracy = 100 * test_correct / test_total
        test_loss = test_loss / len(test_loader)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        # print(f"Test Accuracy after Epoch {epoch + 1}: {test_accuracy}%")
        print(f"Epoch [{epoch + 1}/{epochs}] | Train Loss: {train_loss}, Train Accuracy: {train_accuracy}% | test Loss: {test_loss}, test Accuracy: {test_accuracy}%")

        # 학습 결과 그래프 저장
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, epoch + 2), train_losses, label='Train Loss')
        plt.plot(range(1, epoch + 2), test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(range(1, epoch + 2), train_accuracies, label='Train Accuracy')
        plt.plot(range(1, epoch + 2), test_accuracies, label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'./CAM2/train_CAM2_epoch.png')
        plt.close()

    # 모델 체크포인트 저장
    torch.save(model.state_dict(), './CAM2/final_model_checkpoint.pth')
else: 
    checkpoint = torch.load('./CAM2/final_model_checkpoint.pth')
    model.load_state_dict(checkpoint)
    
# CAM 시각화를 위한 후처리 함수
def generate_cam(model, image_tensor, target_class):
    model.eval()
    with torch.no_grad():
        features = model.features(image_tensor.unsqueeze(0).to(device))
        weights = model.fc.weight[target_class].unsqueeze(1).unsqueeze(2)
        cam = torch.sum(features * weights, dim=1).squeeze(0)
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        cam = cam * 255
        cam_resized = cv2.resize(cam, (image_tensor.shape[2], image_tensor.shape[1]))  # 원래 이미지 크기로 resize
    return cam_resized

# CAM 생성 및 시각화 코드
def visualize_cam(model, image_tensor, target_class, save_path):
    cam = generate_cam(model, image_tensor, target_class)
    plt.figure(figsize=(18, 9))
    
    image_tensor = (image_tensor - image_tensor.min())/(image_tensor.max() - image_tensor.min())
    image_tensor = image_tensor * 255

    # 원본 이미지
    plt.subplot(1, 3, 1)
    plt.imshow(image_tensor.permute(1, 2, 0).cpu().numpy())
    plt.title('Original Image')
    plt.axis('off')

    # CAM 이미지
    plt.subplot(1, 3, 2)
    plt.imshow(image_tensor.permute(1, 2, 0).cpu().numpy())
    plt.imshow(cam, cmap='jet', alpha=0.5)
    plt.title('Image + CAM')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    # plt.imshow(image_tensor.permute(1, 2, 0).cpu().numpy())
    plt.imshow(cam, cmap='jet')
    plt.title('CAM Visualization')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# CAM 시각화 예시
sample_image, sample_label = dataset[image_index]
visualize_cam(model, sample_image, sample_label, './CAM2/sample_cam.png')
