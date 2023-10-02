from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

res18_preprocess = ResNet18_Weights.DEFAULT.transforms()


class Res18(nn.Module):
    def __init__(self, num_of_category):
        super().__init__()
        self.backbone = resnet18(weights=None)
        self.backbone.fc = nn.Linear(512, num_of_category)
        self.backbone.train()

    def forward(self, x):
        x = self.backbone(x)
        return x
