import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


device = 'cuda' if torch.cuda.is_available() else 'cpu'

#################################################################################################
model_train = True
test_img_path = './examples/cat1.jpg'

#################################################################################################


# Define an improved CNN model
class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.fc = nn.Linear(256, 37)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.gap(x)  # Apply GAP
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Instantiate and train the model
model = ImprovedCNN()
model.to(device)

# Set the model to training mode
model.train()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Load training data (CIFAR-10 as an example)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_dataset = datasets.OxfordIIITPet(root='./data', split='trainval', download=True, transform=transform)
test_dataset = datasets.OxfordIIITPet(root='./data', split='test', download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Train the model for multiple epochs
if model_train:
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Evaluate on test data
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Test Accuracy: {accuracy:.2f}%")

    # save model
    checkpoints = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoints, './cam_gap_checkpoint.pth')
else:
    checkpoint = torch.load('./cam_gap_checkpoint.pth')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optim'])
    epoch = checkpoint['epoch']


# Set the model to evaluation mode
model.eval()

# Choose the layer for CAM (last convolutional layer)
final_conv_layer = model.conv4

# Register a hook to capture the feature maps from the selected layer
feature_maps = []
def hook_fn(module, input, output):
    feature_maps.append(output.detach())

hook_handle = final_conv_layer.register_forward_hook(hook_fn)

# Load an image and preprocess it
image_path = "./examples/cat1.jpg"  # Replace with your image path
image = Image.open(image_path)

input_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = input_transform(image).unsqueeze(0).to(device)

# Forward pass through the model
outputs = model(input_tensor)

# Get the class with the highest score
predicted_class = outputs.argmax(dim=1).item()

# Get the weight of the fully connected layer corresponding to the predicted class
params = list(model.parameters())
fc_weights = params[-2].detach()[predicted_class]

# Get the feature maps from the hooked layer
feature_map = feature_maps[0].squeeze()

fc_weights = fc_weights.to('cpu')
feature_map = feature_map.to('cpu')

# Calculate the CAM by taking a weighted sum of the feature maps
cam = torch.zeros(feature_map.shape[1:], dtype=torch.float32)
for i, w in enumerate(fc_weights):
    cam += w * feature_map[i]

# Apply ReLU to keep positive contributions only
cam = F.relu(cam)

# Normalize the CAM for visualization
cam -= cam.min()
cam /= cam.max()
cam = cam.numpy()

# Resize the CAM to the original image size
cam = np.uint8(255 * cam)
cam = Image.fromarray(cam).resize(image.size, resample=Image.BILINEAR)
cam = np.array(cam)

# Superimpose the CAM on the original image
image_np = np.array(image)
plt.imshow(image_np)
plt.imshow(cam, cmap='jet', alpha=0.5)
plt.axis('off')
plt.colorbar(label='CAM Intensity')
plt.show()

# Remove the hook
hook_handle.remove()
