from .resnet import resnet50, resnet101
import torch.nn as nn
import torch
from .globalNet import globalNet
from .refineNet import refineNet
from .segNet import segNet
from .multiNet import multiNet

__all__ = ['CPN50', 'CPN101']

class CPN(nn.Module):
    def __init__(self, resnet, output_shape, num_class, pretrained=True):
        super(CPN, self).__init__()
        channel_settings = [2048, 1024, 512, 256]
        self.resnet = resnet
        self.globalNet = globalNet(channel_settings, output_shape)
        self.multiNet = multiNet(channel_settings[3], output_shape)

        # self.global_net = globalNet(channel_settings, output_shape, num_class)
        # self.refine_net = refineNet(channel_settings[-1], output_shape, num_class)


    def forward(self, x):
        # import pdb;pdb.set_trace()
        res_out = self.resnet(x)
        # import pdb;pdb.set_trace()
        global_out = self.globalNet(res_out)

        end_point_pred, intersection_point_pred, end_points_short_offsets_pred, intersection_points_short_offsets_pred = self.multiNet(global_out)

        # global_fms, global_outs, global_offset = self.global_net(res_out)
        # refine_out = self.refine_net(global_fms)
        # seg_out = self.seg_net(res_out)

        # import pdb; pdb.set_trace()
        return end_point_pred, intersection_point_pred, end_points_short_offsets_pred, intersection_points_short_offsets_pred

def CPN50(out_size,num_class,pretrained=True):
    res50 = resnet50(pretrained=pretrained)
    model = CPN(res50, output_shape=out_size,num_class=num_class, pretrained=pretrained)
    return model

def CPN101(out_size,num_class,pretrained=True):
    res101 = resnet101(pretrained=pretrained)
    model = CPN(res101, output_shape=out_size,num_class=num_class, pretrained=pretrained)
    return model


# class CPN(nn.Module):
#     def __init__(self, resnet, output_shape, num_class, pretrained=True):
#         super(CPN, self).__init__()
#         channel_settings = [2048, 1024, 512, 256]
#         self.resnet = resnet
#         self.seg_net = segNet(channel_settings, output_shape, num_class = 1)
#         self.global_net = globalNet(channel_settings, output_shape, num_class)
#         self.refine_net = refineNet(channel_settings[-1], output_shape, num_class)


#     def forward(self, x):
#         res_out = self.resnet(x)
#         global_fms, global_outs, global_offset = self.global_net(res_out)
#         refine_out = self.refine_net(global_fms)
#         seg_out = self.seg_net(res_out)

#         # import pdb; pdb.set_trace()
#         return global_outs,refine_out, global_offset, seg_out

# def CPN50(out_size,num_class,pretrained=True):
#     res50 = resnet50(pretrained=pretrained)
#     model = CPN(res50, output_shape=out_size,num_class=num_class, pretrained=pretrained)
#     return model

# def CPN101(out_size,num_class,pretrained=True):
#     res101 = resnet101(pretrained=pretrained)
#     model = CPN(res101, output_shape=out_size,num_class=num_class, pretrained=pretrained)
#     return model
