import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

import torch.nn.functional as F

import torch.optim as optim

import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np

#############
torch.manual_seed(42)
transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.5], [0.5])]
)

batch_size = 64

trainset = torchvision.datasets.FashionMNIST(
    root=".", train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True
)

images, labels = next(iter(trainloader))
grid = make_grid(0.5-images/2, 8, 4)
plt.imshow(grid.numpy().transpose((1, 2, 0)),
           cmap = "gray_r")
plt.axis("off")
plt.show()

#################
class Generator(nn.Module):
    
    def __init__(self):
        super(Generator, self).__init__()
        self.z_dim = 100
        self.model = nn.Sequential(
            nn.Linear(self.z_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 784),
            nn.Tanh()
        )
    
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 784)
        return img

class Discriminator(nn.Module):
    
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
                nn.Linear(784, 1024),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3),
                nn.Linear(1024, 512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, 1),
                nn.Sigmoid()
        )
    
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

device = "cuda" if torch.cuda.is_available() else "cpu"
generator = Generator().to(device)
discriminator = Discriminator().to(device)

#################
criterion = nn.BCELoss()

g_optimizer = optim.Adam(generator.parameters(), lr=1e-4)
d_optimizer = optim.Adam(discriminator.parameters(), lr=1e-4)

#################
def show_generated_images(images, num_images=25, title='Generated Images'):
    plt.figure(figsize=(10, 10))
    images = images / 2 + 0.5
    images = images.clamp(0, 1)
    image_grid = make_grid(images[:num_images], nrow=5, normalize=False).cpu().numpy()
    plt.imshow(np.transpose(image_grid, (1, 2, 0)), interpolation='nearest')
    plt.title(title)
    plt.axis('off')
    plt.show()

num_epochs = 50
for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(trainloader):
        current_batch_size = real_images.size(0)
        real_images = real_images.to(device)
        
        real_labels = torch.ones(current_batch_size, 1).to(device)
        fake_labels = torch.zeros(current_batch_size, 1).to(device)
        
        d_optimizer.zero_grad()
        
        real_outputs = discriminator(real_images)
        d_loss_real = criterion(real_outputs, real_labels)
        
        z = torch.randn(current_batch_size, 100).to(device)
        fake_images = generator(z)
        fake_outputs = discriminator(fake_images.detach())
        d_loss_fake = criterion(fake_outputs, fake_labels)
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        d_optimizer.step()
        
        g_optimizer.zero_grad()
        
        fake_outputs = discriminator(fake_images)
        g_loss = criterion(fake_outputs, real_labels)
        
        g_loss.backward()
        g_optimizer.step()
        
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(trainloader)}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}')
        if i+1 == len(trainloader):
                show_generated_images(fake_images.view(-1, 1, 28, 28), title=f'Epoch {epoch+1}')
            
print("Training finished")