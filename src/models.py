
import timm
from torch import nn


class pretrainedModel(nn.Module):
    def __init__(self, model_arc='resnet18d', num_classes=18):
        super().__init__()
        self.net = timm.create_model('swin_base_patch4_window7_224', pretraind=True, num_classes=num_classes)
    
    def forward(self,x):
        x=self.net()

        return x