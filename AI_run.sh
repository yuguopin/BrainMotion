source activate
conda activate torchGPU

for model in 'CNN' 'CNNPLus' 'BrainEmotion'; do
    for lr in 0.1 0.001 0.0005 0.0001; do
        for epoch in 10 20 30 40; do
            for dataset in 'CIFAR100' 'FashionMNIST'; do
                python main.py --model $model --lr $lr --epoch $epoch --dataset $dataset
            done
        done
    done
done
