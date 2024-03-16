# source activate
# conda activate torchGPU

for model in 'CNN' 'CNNPLus' 'BrainEmotion'; do
    for lr in 0.1 0.001 0.0005 0.0001; do
        for epoch in 30 50; do
            for dataset in 'CIFAR100' 'FashionMNIST'; do
                python main.py --model $model --lr $lr --epoch $epoch --dataset $dataset --device cuda:4
            done
        done
    done
done
