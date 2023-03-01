import matplotlib.pyplot as plt
import numpy as np
import torchvision
from pytorch_adapt.datasets import get_office31

# root="datasets/pytorch-adapt/"


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

inv_normalize = torchvision.transforms.Normalize(
    mean=[-m / s for m, s in zip(mean, std)], std=[1 / s for s in std]
)

idx = 0

def imshow(img, domain, figsize=(10, 6)):
    img = inv_normalize(img)
    npimg = img.numpy()
    plt.figure(figsize=figsize)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')

    plt.savefig(f"office31-{idx}")
    plt.show()
    plt.close("all")
    idx += 1

def imshow_many(datasets, src, target):
    d = datasets["train"]
    for name in ["src_imgs", "target_imgs"]:
        domains = src if name == "src_imgs" else target
        if len(domains) == 0:
            continue
        print(domains)
        imgs = [d[i][name] for i in np.random.choice(len(d), size=16, replace=False)]
        imshow(torchvision.utils.make_grid(imgs))

for src, target in [(["amazon"], ["dslr"]), (["webcam"], [])]:
    datasets = get_office31(src, target,folder=root)
    imshow_many(datasets, src, target)
