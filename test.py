import torch
from torchaudio import models, transforms
import torchvision

import matplotlib.pyplot as plt

transforms = torchvision.transforms.Compose(
    [torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)
img_tensor = torchvision.io.read_image("./elijah.png", mode=torchvision.io.ImageReadMode.RGB)
normalized_img_tensor = transforms(img_tensor.type(torch.float))
print(img_tensor.shape)

plt.imshow(img_tensor.permute(1,2,0))
plt.imshow(normalized_img_tensor.permute(1,2,0))
plt.show()