import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim

transform = transforms.ToTensor()

#trainset = datasets.MNIST('./MNIST_data/', dowload=True, train=True, transform=transform)
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

#valset = datasets.MNIST('./MNIST_data/', dowload=True, train=True, transform=transform)
#valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

dataiter = iter(trainloader)
imagens, etiquetas = dataiter.next()
plt.imshow(imagens[0].numpy().squeeze(), cmap='gray_r');

print(imagens[0].shape)
print(etiquetas[0].shape)

class modelo(nn.Module):
    super(Modelo, self).__init__()
    self.linear1 = nn.Linear(28*28, 128)
    self.linear2 = nn.Linear(128, 64)
    self.linear3 = nn.Linear(64, 10)

def forward(self,X):
    X = F.relu(self.linear1(X))
    X = F.relu(self.linear2(X))
    x = self.linear3(X)
    return F.log_softmax(x, dim=1)

def treino(modelo, trainloader, device):
    otimizador = optim.SGD(modelo.parameters(), Ir=0.01, momentum=0.5)
    inicio = time()

    criterio = nn.NLLLoss()
    EPOCHS = 30
    modelo.train()

for epoch in range(EPOCHS)
perda_acumulada = 0

for imagens, etiquetas in trainloader:
        imagens = imagens.view(imagens.shape[0], -1)
        otimizador.zero_grad()

        output = modelo(imagens.to(device))
        perda_instantanea = criterio(output, etiquetas.to(device))

        perda_instantanea.backward()

        otimizador.step()

        perda_acumulada += perda_instantanea.item()

else:
    print('Epoch {} - Perda resultante: {}'.format(epoch+1,perda_acumulativa/len(trainloader)))

print('\nTempo de treino (em muitos) =',(time()-inicio)/60)

modelo = modelo()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
modelo.to(device)
