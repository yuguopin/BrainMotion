for model in 'BrainEmotion' 'CNN' 'CNNPlus'; do
    for lr in 0.0008 0.0003; do
        for epoch in 30 50; do
            for dataset in 'CIFAR100' 'FashionMNIST'; do
                python main.py --model $model --lr $lr --epoch $epoch --dataset $dataset --device cuda:6
            done
        done
    done
done
