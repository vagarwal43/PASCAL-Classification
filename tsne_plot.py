import torch
import torchvision
from torchvision import models
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from simple_cnn import SimpleCNN
from voc_dataset import VOCDataset 
import utils
from utils import ARGS 
import torch.nn as nn
from train_q2 import ResNet
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from torch.utils.data import Subset

# Part E i.e., the TSNE for question 2


class ResnetTSNE(ResNet):
        def __init__(self, num_classes) -> None:
            super(ResnetTSNE, self).__init__(num_classes = num_classes)
            self.resnet = torchvision.models.resnet18(weights='IMAGENET1K_V1')
            self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
            self.fc1 = nn.Linear(512,num_classes)
            self.fc = nn.Identity()
        
        def forward(self, x):
        ##################################################################
        # TODO: Return raw outputs here

            features = self.resnet(x)
            features = features.view(features.size(0), -1)  # Flatten the features
            output = self.fc1(features)
            output = self.fc(output)
        ##################################################################
            return output


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
checkpoint_path = '/home/dell/hw1/checkpoint-model-epoch15.pth'
num_classes = len(VOCDataset.CLASS_NAMES)
loaded_model = torch.load(checkpoint_path)
model = ResnetTSNE(num_classes).to(device)
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(loaded_model.state_dict())
model.eval()

# Load PASCAL VOC test set
test_dataset = VOCDataset(split='test', size=(224, 224))

# Randomly get 1000 images from test set
indices = np.random.choice(len(test_dataset), 1000, replace=False)
selected_data = Subset(test_dataset, indices)

# Create a DataLoader for the selected data
data_loader = DataLoader(selected_data, batch_size=32, shuffle=False)  # Batch size can be adjusted

# Extracting features
features = []
labels = []
with torch.no_grad():
    for data, target, _ in data_loader:
        data = data.to(device)
        output = model(data)
        features.append(output.cpu().numpy())
        labels.extend(target.cpu().numpy())

# t-SNE projection
features = np.vstack(features)
tsne = TSNE(n_components=2, random_state=42)  
projection = tsne.fit_transform(features)

class_labels = VOCDataset.CLASS_NAMES
colors = plt.cm.get_cmap('tab20', len(class_labels)).colors  

# Plot t-SNE in 2D
plt.figure(figsize=(15, 8))

# Iterate over all class labels
for i, class_label in enumerate(class_labels):
    # For each class, find samples that are labeled with that class
    indices = [j for j, label in enumerate(labels) if label[i] == 1]
    # Scatter each class with its unique color and label
    plt.scatter(projection[indices, 0], projection[indices, 1], color=np.array(colors[i]), label=class_label, alpha=0.5)


plt.title('t-SNE Projection of ImageNet Features')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Save the t-SNE plot to a separate file
plt.savefig('tsne_projection.png')
plt.show()