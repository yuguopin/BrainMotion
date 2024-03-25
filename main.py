import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import argparse
from datetime import datetime

from model import MyConvNet, BrainEmotion, MyConvNetPlus
from utils import train_model
from my_dataset import train_loader_FashionMNIST, test_data_x_FashionMNIST, test_data_y_FashionMNIST, class_label_FashionMNIST
from my_dataset import train_loader_CIFAR100, test_loader_CIFAR100, class_label_CIFAR100
from my_resnet import resnet34


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Some hyperparameters')

    parser.add_argument('--model', type=str, choices=['CNN', 'BrainEmotion', 'CNNPlus', 'resnet34', 'resnet34Brain'], default='resnet34Brain')
    parser.add_argument('--lr', type=float, default='0.001')
    parser.add_argument('--epoch', type=int, default='10')
    parser.add_argument('--device', type=str, default='cuda:4')
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
        test_loader = test_loader_CIFAR100
        # test_data_x = test_data_x_CIFAR100
        # test_data_y = test_data_y_CIFAR100
        class_label = class_label_CIFAR100
    else:
        pass
    
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

    if args.model == 'BrainEmotion':
        myconvnet = BrainEmotion(args=args)
        save_path = f'./res_BrainEmotion/{formatted_time}_{lr}_{epoches}_{args.dataset}'
    elif args.model == 'CNN':
        myconvnet = MyConvNet(args=args)
        save_path = f'./res_CNN/{formatted_time}_{lr}_{epoches}_{args.dataset}'
    elif args.model == 'CNNPlus':
        myconvnet = MyConvNetPlus(args=args)
        save_path = f'./res_CNNPlus/{formatted_time}_{lr}_{epoches}_{args.dataset}'
    elif args.model == 'resnet34':
        myconvnet = resnet34(num_classes=100)
        save_path = f'./resnet34/{formatted_time}_{lr}_{epoches}_{args.dataset}'
    elif args.model == 'resnet34Brain':
        myconvnet = resnet34(num_classes=100, AddBrain=True)
        save_path = f'./resnet34Brain/{formatted_time}_{lr}_{epoches}_{args.dataset}'
    else:
        pass
        
    # print(myconvnet)
    
       
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        

    optimizer = torch.optim.Adam(myconvnet.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss().to(device)
    myconvnet, train_process = train_model(args=args, model=myconvnet, traindataloader=train_loader, criterion=criterion,
                                           optimizer=optimizer, num_epochs=epoches, train_rate=0.8, device=device, save_path=save_path)

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
    
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = myconvnet(torch.squeeze(inputs, 1))
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            print(f'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss / (batch_idx+1), 100.* correct / total, correct, total))
            
    # output = myconvnet(test_data_x.to(device))
    # pre_lab = torch.argmax(output, 1)
    # test_data_ya = test_data_y.detach().numpy()
    # pre_laba = pre_lab.cpu().detach().numpy()
    # acc = accuracy_score(test_data_ya, pre_laba)
    # print("ACC Test:", acc)
    
    acc = 100. * correct / total
    
    with open(f'./{save_path}/test_{lr}_{epoches}_{args.dataset}.txt', 'w') as f:
        f.write(f'Test acc: {acc:.3f}') 

    # plt.figure(figsize=(12, 12))
    # conf_mat = confusion_matrix(test_data_ya, pre_laba)
    # df_cm = pd.DataFrame(conf_mat, index=class_label, columns=class_label)
    # heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="YlGnBu")
    # heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha="right")
    # plt.ylabel('True label')
    # plt.xlabel("Predicted label")
    # plt.savefig(f'./{save_path}/eval_{lr}_{epoches}_{args.dataset}.jpg', dpi=600)
