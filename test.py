from http.client import IM_USED
import traceback
import torch
from torchaudio import datasets, models, transforms
import torchvision

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
# transforms = torchvision.transforms.Compose(
#     [torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
# )
# img_tensor = torchvision.io.read_image("./elijah.png", mode=torchvision.io.ImageReadMode.RGB)
# normalized_img_tensor = transforms(img_tensor.type(torch.float))
# print(img_tensor.shape)

# plt.imshow(img_tensor.permute(1,2,0))
# plt.imshow(normalized_img_tensor.permute(1,2,0))
# plt.show()

# datasets = torchvision.datasets.MNIST(
#     root="./data",
#     train=True,
#     download=True,
#     transform=torchvision.transforms.Compose([
#         torchvision.transforms.ToTensor(),
#     ])
# )

# loader = DataLoader(datasets, batch_size=len(datasets), num_workers=1)
# data = next(iter(loader))
# print(data[0].shape)
# mean = data[0].mean()
# std = data[0].std()



img_tensor1 = torchvision.io.read_image("./elijah.png", mode=torchvision.io.ImageReadMode.RGB)
img_tensor = img_tensor1.type(torch.float)
print(img_tensor.dtype)
mean = img_tensor.mean()
std = img_tensor.std()

transform = torchvision.transforms.Compose([
    torchvision.transforms.Normalize((mean),(std))
])

new = transform(img_tensor)

_, boxes = plt.subplots(1,2)
boxes[0].imshow(img_tensor1.permute(1,2,0))
boxes[1].imshow(new.permute(1,2,0))


plt.show()

