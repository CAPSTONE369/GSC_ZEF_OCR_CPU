"""
Copyright (c) 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import torch
import torch.nn as nn

from .modules.transformation import TPS_SpatialTransformerNetwork
from .modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from .modules.sequence_modeling import BidirectionalLSTM
from .modules.prediction import Attention


class CRNN(nn.Module):

    def __init__(self, recog_config, num_class):
        super(CRNN, self).__init__()
        self.opt = recog_config
        self.num_class = num_class
        self.stages = {'Trans': recog_config['TRANSFORMATION'], 'Feat': recog_config['FEATUREEXTRACTION'],
                       'Seq': recog_config['SEQUENCEMODELING'], 'Pred': recog_config['PREDICTION']}

        """ Transformation """
        if self.opt['TRANSFORMATION'] == 'TPS':
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=self.opt['NUMFIDUCIAL'], I_size=(self.opt['IMGH'], self.opt['IMGW']), \
                    I_r_size=(self.opt['IMGH'], self.opt['IMGW']), I_channel_num=self.opt['INPUTCHANNEL'])
        else:
            print('No Transformation module specified')

        """ FeatureExtraction """
        if self.opt['FEATUREEXTRACTION'] == 'VGG':
            self.FeatureExtraction = VGG_FeatureExtractor(self.opt['INPUTCHANNEL'], self.opt['OUTPUTCHANNEL'])
        elif self.opt['FEATUREEXTRACTION']  == 'RCNN':
            self.FeatureExtraction = RCNN_FeatureExtractor(self.opt['INPUTCHANNEL'], self.opt['OUTPUTCHANNEL'])
        elif self.opt['FEATUREEXTRACTION']  == 'ResNet':
            self.FeatureExtraction = ResNet_FeatureExtractor(self.opt['INPUTCHANNEL'], self.opt['OUTPUTCHANNEL'])
        else:
            raise Exception('No FeatureExtraction module specified')
        self.FeatureExtraction_output = self.opt['OUTPUTCHANNEL']  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        if self.opt['SEQUENCEMODELING'] == 'BiLSTM':
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, self.opt['HIDDENSIZE'], self.opt['HIDDENSIZE']),
                BidirectionalLSTM(self.opt['HIDDENSIZE'], self.opt['HIDDENSIZE'], self.opt['HIDDENSIZE']))
            self.SequenceModeling_output = self.opt['HIDDENSIZE']
        else:
            print('No SequenceModeling module specified')
            self.SequenceModeling_output = self.FeatureExtraction_output

        """ Prediction """
        if self.opt['PREDICTION'] == 'CTC':
            self.Prediction = nn.Linear(self.SequenceModeling_output, self.num_class)
        elif self.opt['PREDICTION'] == 'Attn':
            self.Prediction = Attention(self.SequenceModeling_output, self.opt['HIDDENSIZE'], self.num_class)
        else:
            raise Exception('Prediction is neither CTC or Attn')
        
    def get_device(self):
        return next(self.parameters()).device
    
    def forward(self, input, text, is_train=True):
        """ Transformation stage """
        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        if self.stages['Seq'] == 'BiLSTM':
            contextual_feature = self.SequenceModeling(visual_feature)
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        """ Prediction stage """
        if self.stages['Pred'] == 'CTC':
            prediction = self.Prediction(contextual_feature.contiguous())
        else:
            prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=self.opt.batch_max_length)

        return prediction
