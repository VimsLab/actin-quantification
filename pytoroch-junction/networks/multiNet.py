import torch.nn as nn
import torch
import math

class multiNet(nn.Module):
    def __init__(self, channel_settings, output_shape):
        super(multiNet, self).__init__()
        self.channel_settings = channel_settings
        self.lateral = self._lateral(channel_settings)
        self.predict_segmask = self._predict( num_class = 1)
        self.predict_endpoint = self._predict( num_class = 1)
        self.predict_controlpoint = self._predict( num_class = 1)
        self.predict_short_offset_inter = self._predict( num_class = 2)
        self.predict_short_offset_endpoint = self._predict( num_class = 2)
        self.predict_prev_offset = self._predict( num_class = 2)
        self.predict_next_offset = self._predict( num_class = 2)
        self.predict_long_offset = self._predict( num_class = 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _lateral(self, input_size):
        layers = []
        layers.append(nn.Conv2d(input_size, 256,
            kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def _upsamble(self, input_size, num_class):
        layers = []
        layers.append(nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=True))

    def _predict(self, num_class):
        layers = []
        layers.append(nn.Conv2d(256, 64,
            kernel_size=3, stride=1, padding = 1, bias=False))
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=True))


        # layers.append(nn.Conv2d(64, 32,
        #     kernel_size=3, stride=1, bias=False))
        # layers.append(nn.BatchNorm2d(32))
        # layers.append(nn.ReLU(inplace=True))
        # layers.append(nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=True))


        layers.append(nn.Conv2d(64, num_class,
            kernel_size=1, stride=1, bias=False))
        layers.append(nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=True))
        # layers.append(nn.Upsample(size=output_shape, mode='bilinear', align_corners=True))

        layers.append(nn.BatchNorm2d(num_class))

        return nn.Sequential(*layers)

    def forward(self, x):

        feature = self.lateral(x)
        # import pdb;pdb.set_trace()
        end_point_pred =torch.sigmoid(self.predict_endpoint(feature))
        # import pdb;pdb.set_trace()
        intersection_point_pred =torch.sigmoid(self.predict_controlpoint(feature))


        end_points_short_offsets_pred =self.predict_short_offset_endpoint(feature)
        intersection_points_short_offsets_pred =self.predict_short_offset_inter(feature)

        return end_point_pred, intersection_point_pred, end_points_short_offsets_pred, intersection_points_short_offsets_pred
