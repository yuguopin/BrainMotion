import torch
import torch.nn as nn


class MyConvNet(nn.Module):
    def __init__(self, args):
        super(MyConvNet, self).__init__()
        
        if args.dataset == 'FashionMNIST':
            self.input_channel = 1
            self.fc_dim = 32 * 3 * 3
            self.out_dim = 10
        elif args.dataset == 'CIFAR100':
            self.input_channel = 3
            self.fc_dim = 32 * 4 * 4
            self.out_dim = 100
        else:
            self.input_channel = 0
            self.fc_dim = 0
            self.out_dim = 0
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.input_channel,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.MaxPool2d(2, 2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.MaxPool2d(2, 2),
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.fc_dim, 256), 
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.out_dim)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        output = self.classifier(x)

        return output


class MyConvNetPlus(nn.Module):
    def __init__(self, args):
        super(MyConvNetPlus, self).__init__()
        
        if args.dataset == 'FashionMNIST':
            self.input_channel = 1
            self.fc_dim = 32 * 3 * 3
            self.out_dim = 10
        elif args.dataset == 'CIFAR100':
            self.input_channel = 3
            self.fc_dim = 32 * 3 * 3
            self.out_dim = 100
        else:
            self.input_channel = 0
            self.fc_dim = 0
            self.out_dim = 0
            
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.input_channel,
                out_channels=16,
                kernel_size=1,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 1, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 1, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 32, 1, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.fc_dim, 256), 
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.out_dim)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        output = self.classifier(x)

        return output
    
    
# ==================================================================================
class BrainEmotionConvLayer(nn.Module):
    def __init__(self, input_dim=1, embed_dim=16, out_dim=32) -> None:
        super(BrainEmotionConvLayer, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        
        self.conv1x1_embed = nn.Conv2d(in_channels=self.input_dim, out_channels=self.embed_dim, kernel_size=1)
        self.conv_cortex = nn.Conv2d(in_channels=self.embed_dim, out_channels=self.out_dim, kernel_size=3, padding=1)
        self.conv_body = nn.Conv2d(in_channels=self.embed_dim+1, out_channels=self.out_dim, kernel_size=3, padding=1)
    

    def forward(self, input_data):
        channel_mean = torch.mean(input_data, dim=[0, 2, 3])
        _, max_index = torch.max(channel_mean, dim=0)
        
        max_channel_map = input_data[:, max_index, :, :].unsqueeze(1)

        embed_data = self.conv1x1_embed(input_data)
        # print(f'embed_data.shape: {embed_data.shape}')

        cortex_out = self.conv_cortex(embed_data)
        # print(f'cortex_out.shape: {cortex_out.shape}')

        body_input = torch.cat((embed_data, max_channel_map), dim=1)
        body_out = self.conv_body(body_input)
        # print(f'body_out.shape: {body_out.shape}')

        res = body_out - cortex_out
        # print(f'res.shape: {res.shape}')
        
        return res


class BrainEmotion(nn.Module):
    def __init__(self, args) -> None:
        super(BrainEmotion, self).__init__()
        
        if args.dataset == 'FashionMNIST':
            self.input_channel = 1
            self.fc_dim = 32 * 3 * 3
            self.out_dim = 10
        elif args.dataset == 'CIFAR100':
            self.input_channel = 3
            self.fc_dim = 32 * 4 * 4
            self.out_dim = 100
        else:
            self.input_channel = 0
            self.fc_dim = 0
            self.out_dim = 0
            
        self.conv1 = nn.Sequential(
            BrainEmotionConvLayer(input_dim=self.input_channel, embed_dim=16, out_dim=16),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            BrainEmotionConvLayer(input_dim=16, embed_dim=32, out_dim=16),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            BrainEmotionConvLayer(input_dim=16, embed_dim=32, out_dim=32),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            BrainEmotionConvLayer(input_dim=32, embed_dim=32, out_dim=32),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.MaxPool2d(2, 2),
        )
        
        self.conv3 = nn.Sequential(
            BrainEmotionConvLayer(input_dim=32, embed_dim=64, out_dim=64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            BrainEmotionConvLayer(input_dim=64, embed_dim=64, out_dim=64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.MaxPool2d(2, 2),
        )
        
        self.conv4 = nn.Sequential(
            BrainEmotionConvLayer(input_dim=64, embed_dim=32, out_dim=32),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            BrainEmotionConvLayer(input_dim=32, embed_dim=32, out_dim=32),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.fc_dim, 256), 
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.out_dim)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        output = self.classifier(x)

        return output
        