source activate
conda activate torchGPU

python main.py --model BrainEmotion --lr 0.001 --epoch 30 --dataset 'CIFAR100'
python main.py --model BrainEmotion --lr 0.0005 --epoch 30 --dataset 'CIFAR100'
python main.py --model BrainEmotion --lr 0.0001 --epoch 30 --dataset 'CIFAR100'

python main.py --model BrainEmotion --lr 0.0001 --epoch 10 --dataset 'CIFAR100'
python main.py --model BrainEmotion --lr 0.0001 --epoch 20 --dataset 'CIFAR100'
python main.py --model BrainEmotion --lr 0.0001 --epoch 40 --dataset 'CIFAR100'

python main.py --model BrainEmotion --lr 0.001 --epoch 30 --dataset 'FashionMNIST'
python main.py --model BrainEmotion --lr 0.0005 --epoch 30 --dataset 'FashionMNIST'
python main.py --model BrainEmotion --lr 0.0001 --epoch 30 --dataset 'FashionMNIST'

python main.py --model BrainEmotion --lr 0.0001 --epoch 10 --dataset 'FashionMNIST'
python main.py --model BrainEmotion --lr 0.0001 --epoch 20 --dataset 'FashionMNIST'
python main.py --model BrainEmotion --lr 0.0001 --epoch 40 --dataset 'FashionMNIST'

# ===================================================================================
python main.py --model CNN --lr 0.001 --epoch 30 --dataset 'CIFAR100'
python main.py --model CNN --lr 0.0005 --epoch 30 --dataset 'CIFAR100'
python main.py --model CNN --lr 0.0001 --epoch 30 --dataset 'CIFAR100'

python main.py --model CNN --lr 0.0001 --epoch 10 --dataset 'CIFAR100'
python main.py --model CNN --lr 0.0001 --epoch 20 --dataset 'CIFAR100'
python main.py --model CNN --lr 0.0001 --epoch 40 --dataset 'CIFAR100'

python main.py --model CNN --lr 0.001 --epoch 30 --dataset 'FashionMNIST'
python main.py --model CNN --lr 0.0005 --epoch 30 --dataset 'FashionMNIST'
python main.py --model CNN --lr 0.0001 --epoch 30 --dataset 'FashionMNIST'

python main.py --model CNN --lr 0.0001 --epoch 10 --dataset 'FashionMNIST'
python main.py --model CNN --lr 0.0001 --epoch 20 --dataset 'FashionMNIST'
python main.py --model CNN --lr 0.0001 --epoch 40 --dataset 'FashionMNIST'

# ===================================================================================
python main.py --model CNNPlus --lr 0.001 --epoch 30 --dataset 'CIFAR100'
python main.py --model CNNPlus --lr 0.0005 --epoch 30 --dataset 'CIFAR100'
python main.py --model CNNPlus --lr 0.0001 --epoch 30 --dataset 'CIFAR100'

python main.py --model CNNPlus --lr 0.0001 --epoch 10 --dataset 'CIFAR100'
python main.py --model CNNPlus --lr 0.0001 --epoch 20 --dataset 'CIFAR100'
python main.py --model CNNPlus --lr 0.0001 --epoch 40 --dataset 'CIFAR100'

python main.py --model CNNPlus --lr 0.001 --epoch 30 --dataset 'FashionMNIST'
python main.py --model CNNPlus --lr 0.0005 --epoch 30 --dataset 'FashionMNIST'
python main.py --model CNNPlus --lr 0.0001 --epoch 30 --dataset 'FashionMNIST'

python main.py --model CNNPlus --lr 0.0001 --epoch 10 --dataset 'FashionMNIST'
python main.py --model CNNPlus --lr 0.0001 --epoch 20 --dataset 'FashionMNIST'
python main.py --model CNNPlus --lr 0.0001 --epoch 40 --dataset 'FashionMNIST'
