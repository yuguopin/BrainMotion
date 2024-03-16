import copy
import time
import torch
import pandas as pd


def train_model(args, model, traindataloader, train_rate, criterion, optimizer, num_epochs=25, device=torch.device('cuda:7'), save_path=None):
    model = model.to(device)

    batch_num = len(traindataloader)
    train_batch_num = round(batch_num * train_rate)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []
    since = time.time()

    for epoch in range(num_epochs):
        print('-'*100)

        train_loss = 0.0
        train_corrects = 0
        train_num = 0

        val_loss = 0.0
        val_correct = 0
        val_num = 0

        for step, (b_x, b_y) in enumerate(traindataloader):
            b_x, b_y = b_x.to(device), b_y.to(device)

            if step < train_batch_num:
                model.train()
                output = model(b_x)
                pre_lab = torch.argmax(output, 1)
                loss = criterion(output, b_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_corrects += torch.sum(pre_lab == b_y.data)
                train_num += b_x.size(0)
            else:
                model.eval()
                output = model(b_x)
                pre_lab = torch.argmax(output, 1)
                loss = criterion(output, b_y)
                val_loss += loss.item()
                val_correct += torch.sum(pre_lab == b_y.data)
                val_num += b_y.size(0)

        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item()/train_num)
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_correct.double().item() / val_num)

        print("Epoch: {}/{}\tTrain Loss: {:.4f}\tTrain Acc: {:.4f}".format(epoch + 1, num_epochs, train_loss_all[-1], train_acc_all[-1]))
        print('Epoch: {}/{}\tVal Loss: {:.4f}\tval Acc: {:.4f}'.format(epoch + 1, num_epochs, val_loss_all[-1], val_acc_all[-1]))

        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), f'./{save_path}/best_{args.model}_{args.dataset}_{args.lr}_{args.epoch}.pt')

        time_use = time.time() - since
        print("Train and val complete in {:.0f}m {:.0f}s".format(time_use // 60, time_use % 60))

    model.load_state_dict(best_model_wts)

    train_process = pd.DataFrame(
        data={"epoch": range(num_epochs),
              "train_loss_all": train_loss_all,
              "val_loss_all": val_loss_all,
              "train_acc_all": train_acc_all,
              "val_acc_all": val_acc_all}
    )

    return model, train_process
