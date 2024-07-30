import torch,pdb
import torchvision
import torch.nn.modules

class vgg16bn(torch.nn.Module):
    def __init__(self,pretrained = False):
        super(vgg16bn,self).__init__()
        model = list(torchvision.models.vgg16_bn(pretrained=pretrained).features.children())
        model = model[:33]+model[34:43]
        self.model = torch.nn.Sequential(*model)
        
    def forward(self,x):
        return self.model(x)
class resnet(torch.nn.Module):
    def __init__(self,layers,pretrained = False):
        super(resnet,self).__init__()
        if layers == '18':
            model = torchvision.models.resnet18(pretrained=pretrained)
        elif layers == '34':
            model = torchvision.models.resnet34(pretrained=pretrained)
        elif layers == '50':
            model = torchvision.models.resnet50(pretrained=pretrained)
        elif layers == '101':
            model = torchvision.models.resnet101(pretrained=pretrained)
        elif layers == '152':
            model = torchvision.models.resnet152(pretrained=pretrained)
        elif layers == '50next':
            model = torchvision.models.resnext50_32x4d(pretrained=pretrained)
        elif layers == '101next':
            model = torchvision.models.resnext101_32x8d(pretrained=pretrained)
        elif layers == '50wide':
            model = torchvision.models.wide_resnet50_2(pretrained=pretrained)
        elif layers == '101wide':
            model = torchvision.models.wide_resnet101_2(pretrained=pretrained)
        elif layers == '34fca':
            model = torch.hub.load('cfzd/FcaNet', 'fca34' ,pretrained=True)
        else:
            raise NotImplementedError
        
        # image feature
        self.conv1_img = model.conv1
        self.bn1_img = model.bn1
        self.relu_img = model.relu
        self.maxpool_img = model.maxpool
        
        # depth feature
        self.conv1_depth = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1_depth = torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu_depth = torch.nn.ReLU(inplace=True)
        self.maxpool_depth = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        # after fusion
        self.conv1x1 = torch.nn.Conv2d(128, 64, kernel_size=1)
        self.bn2 = torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = torch.nn.ReLU(inplace=True)

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self,x,y):
        # image feature
        x = self.conv1_img(x)
        x = self.bn1_img(x)
        x = self.relu_img(x)
        x = self.maxpool_img(x)

        # depth feature
        y = self.conv1_depth(y)
        y = self.bn1_depth(y)
        y = self.relu_depth(y)
        y = self.maxpool_depth(y)

        # fusion
        z = torch.cat((x,y),dim=1)
        z = self.conv1x1(z)
        z = self.bn2(z)
        z = self.relu(z)
        
        z1 = self.layer1(z)
        z2 = self.layer2(z1)
        z3 = self.layer3(z2)
        z4 = self.layer4(z3)
        return z2,z3,z4
