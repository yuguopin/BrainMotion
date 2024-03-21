import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST, CIFAR100
import torch
import matplotlib.pyplot as plt
import numpy as np

bs_size = 128

# ======================= CIFAR100 =====================
CIFAR_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_data_CIFAR100 = CIFAR100(root="./data/CIFAR100", train=True, transform=CIFAR_transform, download=True)
train_loader_CIFAR100 = Data.DataLoader(dataset=train_data_CIFAR100, batch_size=bs_size, shuffle=False)

class_label_CIFAR100 = train_data_CIFAR100.classes
# class_label_CIFAR100[0] = "T-shirt"
    
test_data_CIFAR100 = CIFAR100(root="./data/CIFAR100", train=False, transform=CIFAR_transform, download=True)
test_loader_CIFAR100 = Data.DataLoader(dataset=test_data_CIFAR100, batch_size=bs_size, shuffle=False)

# test_data_x_CIFAR100 = torch.from_numpy(test_data_CIFAR100.data / 255.0).float()
# test_data_x_CIFAR100 = test_data_x_CIFAR100.permute(0, 3, 1, 2)
# test_data_y_CIFAR100 = torch.tensor(test_data_CIFAR100.targets)
# print(test_data_y_CIFAR100.shape)

# =============== FashionMNSIT ===================== 
train_data_FashionMNIST = FashionMNIST(root="./data/FashionMNIST", train=True, transform=transforms.ToTensor(), download=True)
train_loader_FashionMNIST = Data.DataLoader(dataset=train_data_FashionMNIST, batch_size=bs_size, shuffle=False)

class_label_FashionMNIST = train_data_FashionMNIST.classes
class_label_FashionMNIST[0] = "T-shirt"

test_data_FashionMNIST = FashionMNIST(root="./data/FashionMNIST", train=False, download=True)

test_data_x_FashionMNIST = test_data_FashionMNIST.data.type(torch.FloatTensor) / 255.0
test_data_x_FashionMNIST = torch.unsqueeze(test_data_x_FashionMNIST, dim=1)
test_data_y_FashionMNIST = test_data_FashionMNIST.targets


if __name__ == '__main__':
    print("Train_loader 的batch数量为：", len(train_loader_FashionMNIST))

    for step, (b_x, b_y) in enumerate(train_loader_FashionMNIST):
        # print(b_x.max())
        if step > 0:
            break

    batch_x = b_x.squeeze().numpy()
    batch_y = b_y.numpy()
    
    plt.figure(figsize=(12, 5))

    # 可视化FashionMnist数据集
    for ii in np.arange(len(batch_y)):
        plt.subplot(4, 16, ii+1)
        plt.imshow(batch_x[ii, :, :], cmap=plt.cm.gray)
        plt.title(class_label_FashionMNIST[batch_y[ii]], size=9)
        plt.axis("off")
        plt.subplots_adjust(wspace=0.05)

    plt.savefig('./res/FashionMnist.jpg', dpi=600)

    print("test_data_x.shape:", test_data_x_FashionMNIST.shape)
    print("test_data_y.shape:", test_data_y_FashionMNIST.shape)
