import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import argparse
from datetime import datetime

from model import MyConvNet, BrainEmotion, MyConvNetPlus
from utils import train_model
from datasets import train_loader_FashionMNIST, test_data_x_FashionMNIST, test_data_y_FashionMNIST, class_label_FashionMNIST
from datasets import train_loader_CIFAR100, test_data_x_CIFAR100, test_data_y_CIFAR100, class_label_CIFAR100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Some hyperparameters')

    parser.add_argument('--model', type=str, choices=['CNN', 'BrainEmotion', 'CNNPlus'], default='CNN')
    parser.add_argument('--lr', type=float, default='0.0001')
    parser.add_argument('--epoch', type=int, default='30')
    parser.add_argument('--device', type=str, default='cuda:7')
    parser.add_argument('--dataset', type=str, default='CIFAR100', choices=['FashionMNIST', 'CIFAR100'])

    args = parser.parse_args()
    print(args)
    # =====================================================
    device = torch.device(args.device)
    lr = args.lr
    epoches = args.epoch
    
    if args.dataset == 'FashionMNIST':
        train_loader = train_loader_FashionMNIST
        test_data_x = test_data_x_FashionMNIST
        test_data_y = test_data_y_FashionMNIST
        class_label = class_label_FashionMNIST
    elif args.dataset == 'CIFAR100':
        train_loader = train_loader_CIFAR100
        test_data_x = test_data_x_CIFAR100
        test_data_y = test_data_y_CIFAR100
        class_label = class_label_CIFAR100
    else:
        pass
    
    
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

    if args.model == 'BrainEmotion':
        myconvnet = BrainEmotion(args=args)
        save_path = f'./res_BrainEmotion/{formatted_time}'
    elif args.model == 'CNN':
        myconvnet = MyConvNet(args=args)
        save_path = f'./res_CNN/{formatted_time}'
    else:
        myconvnet = MyConvNetPlus(args=args)
        save_path = f'./res_CNNPlus/{formatted_time}'
        
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    # print(myconvnet)

    optimizer = torch.optim.Adam(myconvnet.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss().to(device)
    myconvnet, train_process = train_model(args=args, model=myconvnet, traindataloader=train_loader, criterion=criterion,
                                           optimizer=optimizer, num_epochs=epoches, train_rate=0.8, device=device)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_process.epoch, train_process.train_loss_all, "ro-", label="Train_loss")
    plt.plot(train_process.epoch, train_process.val_loss_all, "bs-", label="Val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.title(f'lr={lr} epoch={epoches} dataset={args.dataset}')

    plt.subplot(1, 2, 2)
    plt.plot(train_process.epoch, train_process.train_acc_all, "ro-", label="Train_acc")
    plt.plot(train_process.epoch, train_process.val_acc_all, "bs-", label="Val acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.title(f'lr={lr} epoch={epoches} dataset={args.dataset}')

    plt.savefig(f'./{save_path}/TrainLossAcc_{lr}_{epoches}_{args.dataset}.jpg', dpi=600)

    # ====== 测试模型 =======
    myconvnet.eval()
    output = myconvnet(test_data_x.to(device))
    pre_lab = torch.argmax(output, 1)
    test_data_ya = test_data_y.detach().numpy()
    pre_laba = pre_lab.cpu().detach().numpy()
    acc = accuracy_score(test_data_ya, pre_laba)
    print("ACC Test:", acc)
    
    with open(f'./{save_path}/test_{lr}_{epoches}_{args.dataset}.txt', 'w') as f:
        f.write(f'Test acc: {acc:.3f}') 

    plt.figure(figsize=(12, 12))
    conf_mat = confusion_matrix(test_data_ya, pre_laba)
    df_cm = pd.DataFrame(conf_mat, index=class_label, columns=class_label)
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="YlGnBu")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha="right")
    plt.ylabel('True label')
    plt.xlabel("Predicted label")
    plt.savefig(f'./{save_path}/eval_{lr}_{epoches}_{args.dataset}.jpg', dpi=600)
