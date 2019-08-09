import torch
import torch.nn as nn
import torch.functional as F

from functions.functions import DecoderBlock
import torchvision


class FCN(nn.Module):
    def __init__(self, is_deconv, **kwargs):
        super().__init__()
        resnet = torchvision.models.resnet34(pretrained=True)       
      
        decoder_inputs = [512, 512+256, 256+256, 256+128, 64+64,128]
        decoder_outputs= [256, 256,256, 64, 128, 32]

        self.pool = nn.MaxPool2d(2,2)
        
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.decoder4 = DecoderBlock(
                        decoder_inputs[0],
                        decoder_inputs[0] // 4,
                        decoder_outputs[0],
                        is_deconv=is_deconv
                    )
 
        self.decoder3 = DecoderBlock(
                        decoder_inputs[1],
                        decoder_inputs[1] // 4,
                        decoder_outputs[1],
                        is_deconv=is_deconv
                    ) 

        self.decoder2 = DecoderBlock(
                        decoder_inputs[2],
                        decoder_inputs[2] // 4,
                        decoder_outputs[2],
                        is_deconv=is_deconv
                    )

        self.decoder1 = DecoderBlock(
                        decoder_inputs[3],
                        decoder_inputs[3] // 4,
                        decoder_outputs[3],
                        is_deconv=is_deconv
                    )

        self.last2decoder = DecoderBlock(
                        decoder_inputs[4],
                        decoder_inputs[4] // 4,
                        decoder_outputs[4],
                        is_deconv=is_deconv
                    )

        self.last1decoder = DecoderBlock(
                        decoder_inputs[5],
                        decoder_inputs[5] // 4,
                        decoder_outputs[5],
                        is_deconv=is_deconv
                    )

        self.final2conv = nn.Conv2d(decoder_outputs[5], decoder_outputs[5], kernel_size=3, stride=3)   
        self.final1conv = nn.Conv2d(decoder_outputs[5], 1, kernel_size=1, stride=1)

    def forward(self, t):
        conv1 = self.conv1(t)
        bn1 = self.bn1(conv1)
        relu = self.relu(bn1)
        maxpool = self.maxpool(relu)
        encoder1 = self.encoder1(maxpool)
        encoder2 = self.encoder2(encoder1)
        encoder3 = self.encoder3(encoder2)
        encoder4 = self.encoder4(encoder3)
        pool = nn.MaxPool2d(2,2)(encoder4)
        
        block4 = torch.cat((self.decoder4(pool), encoder4), 1) 
        block3 = torch.cat((self.decoder3(block4), encoder3), 1)
        block2 = torch.cat((self.decoder2(block3), encoder2), 1)
        block1 = torch.cat((self.decoder1(block2), encoder1), 1)
        
        last2decoder = self.last2decoder(block1)
        last1decoder = self.last1decoder(last2decoder)
        
        final2conv = self.final2conv(last1decoder)
        final1conv = self.final1conv(final2conv)

        return F.sigmoid(final1conv)
