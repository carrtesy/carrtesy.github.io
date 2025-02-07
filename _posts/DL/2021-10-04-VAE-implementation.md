---
title: "Implementing VAE"
date: 2021-10-04
last_modified_at: 2021-10-04
categories:
 - Deep Learning 

tags:
 - Deep Learning
 - Generative Models
 - VAE
 - AE
use_math: true
---
Let's implement VAE using MNIST dataset!

## Load MNIST

Torch has MNIST dataset support. Let's use them.


```python
import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
```


```python
mnist_train = datasets.MNIST(root='MNIST_data/',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)

mnist_test = datasets.MNIST(root='MNIST_data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)
```

```text
Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to MNIST_data/MNIST/raw/train-images-idx3-ubyte.gz

```


```text
  0%|          | 0/9912422 [00:00<?, ?it/s]
```


```text
Extracting MNIST_data/MNIST/raw/train-images-idx3-ubyte.gz to MNIST_data/MNIST/raw

Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to MNIST_data/MNIST/raw/train-labels-idx1-ubyte.gz

```


```text
  0%|          | 0/28881 [00:00<?, ?it/s]
```


```text
Extracting MNIST_data/MNIST/raw/train-labels-idx1-ubyte.gz to MNIST_data/MNIST/raw

Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to MNIST_data/MNIST/raw/t10k-images-idx3-ubyte.gz

```


```text
  0%|          | 0/1648877 [00:00<?, ?it/s]
```


```text
Extracting MNIST_data/MNIST/raw/t10k-images-idx3-ubyte.gz to MNIST_data/MNIST/raw

Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to MNIST_data/MNIST/raw/t10k-labels-idx1-ubyte.gz

```


```text
  0%|          | 0/4542 [00:00<?, ?it/s]
```


```text
Extracting MNIST_data/MNIST/raw/t10k-labels-idx1-ubyte.gz to MNIST_data/MNIST/raw


```

Let's setup dataloaders. 


```python
BATCH_SIZE = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
```

```text
cuda

```


```python
from torch.utils.data import DataLoader

dataloaders = {}
dataloaders["train"] = DataLoader(dataset=mnist_train,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True,
                                          drop_last=True)
```

## Model Implementation

VAE has encoder and decoder as its component, which is the same architecture as AE. However, the difference lies in sampling: μ and σ !

Encoder has two outputs, *mu* (μ) and *log_var* ($logσ^2$). 
The reason for taking $logσ^2$ rather than $σ$ itself is quite clear. 

Encoder output may produce any real number, so it is not guaranteed to be positive. We need $σ$ to be bigger than $0$. 

Using those techniques, encoder produces $\mathcal{N}(μ, σ)$.

Decoder gets latent $z$ ~ $ \mathcal{N}(μ, σ)$, and this $z$ reconstructs image. 


```python
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(32, 10)
        self.fc_log_var = nn.Linear(32, 10)

        self.decoder = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
        )

    def forward(self, x):
        x = x.reshape(-1, 1*28*28)
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        out = self.decoder(z) 
        out = out.reshape(-1, 1, 28, 28)
        return out, mu, log_var

    def encode(self, x):
        h = self.encoder(x)
        mu, log_var = self.fc_mu(h), self.fc_log_var(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        eps = torch.randn(mu.shape).to(mu.device)
        std = torch.exp(0.5 * log_var)
        z = mu + eps*std
        return z

    def decode(self, x):
        return self.decoder(x)
```

However, model itself does not have a power two generate images.
Loss term has two parts: $KLD$ and $Reconstruction$. 

1. $KLD$ forces encoder's outputs to follow certain distribution, namely $\mathcal{N}(0, I)$.

2. $Reconstruction$ minimizes difference of input and its decoded output. 


```python
from torch import optim
import torch.nn.functional as  F
def loss_fn(x, recon_x, mu, log_var):
    Recon_loss = F.mse_loss(recon_x.view(-1, 784), x.view(-1, 784), reduction = "sum")
    KLD_loss = 0.5 * torch.sum(mu.pow(2) + log_var.exp() - 1 - log_var)
    return Recon_loss + KLD_loss


model = VAE()
model.to(device)
optimizer = optim.Adam(model.parameters(), lr = 1e-03)
```

## Train!


```python
epoch = 10

from tqdm import tqdm

for e in range(1, epoch+1):

    train_loss = 0.0
    for x, _ in tqdm(dataloaders["train"]):
        x = x.to(device)
        x_recon, mu, log_var = model(x)

        optimizer.zero_grad()
        loss = loss_fn(x, x_recon, mu, log_var)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    train_loss /= len(dataloaders["train"].dataset)

    print(f"EPOCH {e}: {train_loss}")
```

```text
100%|██████████| 937/937 [00:15<00:00, 60.56it/s]

```

```text
EPOCH 1: 46.222997973632815

```

```text
100%|██████████| 937/937 [00:12<00:00, 73.47it/s]

```

```text
EPOCH 2: 36.6455737508138

```

```text
100%|██████████| 937/937 [00:12<00:00, 74.22it/s]

```

```text
EPOCH 3: 34.71712950642904

```

```text
100%|██████████| 937/937 [00:12<00:00, 73.54it/s]

```

```text
EPOCH 4: 33.845694342041014

```

```text
100%|██████████| 937/937 [00:13<00:00, 71.19it/s]

```

```text
EPOCH 5: 33.33422303466797

```

```text
100%|██████████| 937/937 [00:12<00:00, 72.83it/s]

```

```text
EPOCH 6: 32.95128912760417

```

```text
100%|██████████| 937/937 [00:13<00:00, 70.49it/s]

```

```text
EPOCH 7: 32.67411953735352

```

```text
100%|██████████| 937/937 [00:12<00:00, 73.87it/s]

```

```text
EPOCH 8: 32.436136362711586

```

```text
100%|██████████| 937/937 [00:12<00:00, 74.13it/s]

```

```text
EPOCH 9: 32.26390340779622

```

```text
100%|██████████| 937/937 [00:12<00:00, 73.89it/s]
```

```text
EPOCH 10: 32.058827408854164

```

```text


```

## Generate!

Let's try generating some images using our trained VAE. 


```python
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np

def to_img(x):
    x = x.clamp(0, 1)
    return x

def show_image(img):
    img = to_img(img)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
```

To do so, let's sample $z$ ~ $\mathcal{N}(0, I)$ and feed that as decoder's input.

*torch.randn* is a way to do so.


```python
with torch.no_grad():

    # sample latent vectors from the normal distribution
    latent = torch.randn(128, 10, device=device)

    # reconstruct images from the latent vectors
    img_recon = model.decoder(latent)
    img_recon = img_recon.cpu()

    fig, ax = plt.subplots(figsize=(10, 10))
    imgs = img_recon.reshape(-1, 1, 28, 28).data[:100]
    

    show_image(make_grid(imgs,10,5))
    plt.show()
```

![VAE](..\..\assets\images\AE_VAE\VAE_generated.png)
