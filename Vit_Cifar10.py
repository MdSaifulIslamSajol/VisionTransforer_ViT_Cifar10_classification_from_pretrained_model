 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 12:38:03 2022

@author: saiful
"""

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from transformers import ViTForImageClassification
import torch
from PIL import Image
import requests
from transformers import ViTFeatureExtractor
from tqdm import tqdm
import time 

transform = transforms.Compose(
    [transforms.ToTensor()])


batch_size = 100

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=True, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# select a  single image from test set testset[i][0]
im = testset[1][0]

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
imshow(im)
def get_device():

    if torch.cuda.is_available():
        _device = torch.device('cuda:3') 
    else:
        _device = 'cpu'
    return _device
device = get_device() 

# =============================================================================
    #  Vit
# =============================================================================
model = ViTForImageClassification.from_pretrained('nateraw/vit-base-patch16-224-cifar10')
model.eval()
model.to(device)

def predict_class(img):
    
    feature_extractor = ViTFeatureExtractor.from_pretrained('nateraw/vit-base-patch16-224-cifar10')
    encoding = feature_extractor(images=img, return_tensors="pt")
    encoding.keys()
    encoding['pixel_values'].shape   # torch.Size([1, 3, 224, 224])
    pixel_values = encoding['pixel_values'].to(device)

    outputs = model(pixel_values)
    logits = outputs.logits
    logits.shape        # torch.Size([1, 1000])
    prediction = logits.argmax(dim=1)   
    return prediction.item()

correct = 0
total = 0
with torch.no_grad():
    
     for batch_idx, (inputs, targets) in (enumerate(tqdm(testloader))):
        inputs, targets = inputs.to(device), targets.to(device)
        
        for id, image  in enumerate(inputs):            
            label1=targets[id]
            prediction1 = torch.tensor(predict_class(image.cpu()))
            prediction1=prediction1.to(device)            
            total += 1  #label1.clone().detach()            
            correct += (prediction1 == label1).sum().item()
        acc=(100 * correct / total)
        print( f"\nrunning accuracy:{acc} %")

print (f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


































































