import torch
import torch.nn as nn
from config import cfg

def get_losses(ground_truth, outputs):

    endpoints_target, intersections_points_target, end_points_short_offsets_target, intersection_points_short_offsets_target =  ground_truth
    end_point_pred, intersection_point_pred, end_points_short_offsets_pred, intersection_points_short_offsets_pred = outputs


    loss_end_pt = kp_map_loss(endpoints_target, end_point_pred)
    loss_inter_pt = kp_map_loss(intersections_points_target, intersection_point_pred)


    loss_short_end_pt = short_offset_loss(end_points_short_offsets_target, end_points_short_offsets_pred)

    loss_short_inter_pt = short_offset_loss(intersection_points_short_offsets_target, intersection_points_short_offsets_pred)

    # losses = 4 * (loss_inter_pt + loss_end_pt) + loss_short_end_pt + 2 * loss_short_inter_pt
    losses = 2 * (loss_inter_pt + loss_end_pt) + loss_short_end_pt + 1 * loss_short_inter_pt
    return losses, loss_end_pt, loss_inter_pt, loss_short_end_pt, loss_short_inter_pt

def mask_loss(mask_true, mask_pred):
    criterion_bce = torch.nn.BCELoss().to(cfg.device)

    loss = criterion_bce(mask_pred, mask_true)
    loss = torch.mean(loss)
    return loss

def kp_map_loss(kp_maps_true, kp_maps_pred):
    criterion_bce = torch.nn.BCELoss().to(cfg.device)
    loss = criterion_bce(kp_maps_pred, kp_maps_true)
    loss = torch.mean(loss)
    return loss


def short_offset_loss(short_offset_true, short_offsets_pred):
    kp_maps_true = (short_offset_true!=0).type(torch.float32)
    short_offset_true = short_offset_true * kp_maps_true
    short_offsets_pred = short_offsets_pred * kp_maps_true


    criterion_abs = torch.nn.L1Loss(reduction='sum').to(cfg.device)
    loss = criterion_abs(short_offset_true, short_offsets_pred)/cfg.disc_radius * 1.0

    loss = loss / (torch.sum(kp_maps_true)+1)
    return loss


def mid_offset_loss(mid_offset_true, mid_offset_pred):
    kp_maps_true = (mid_offset_true!=0).type(torch.float32)
    mid_offset_true = mid_offset_true * kp_maps_true
    mid_offset_pred = mid_offset_pred * kp_maps_true

    criterion_abs = torch.nn.L1Loss(reduction='sum').to(cfg.device)
    loss = criterion_abs(mid_offset_true, mid_offset_pred)/cfg.disc_radius * 1.0

    loss = loss / (torch.sum(kp_maps_true)+1)
    return loss


def long_offset_loss(long_offset_true, long_offset_pred):
    criterion_abs = torch.nn.L1Loss(reduction='sum').to(cfg.device)
    seg_true = (long_offset_true!=0).type(torch.float32)

    long_offset_true = long_offset_true * seg_true
    long_offset_pred = long_offset_pred * seg_true


    loss = criterion_abs(long_offset_true, long_offset_pred)/cfg.disc_radius * 1.0
    loss = loss / (torch.sum(seg_true)+1)

    return loss


def segmentation_loss(seg_true, seg_pred):
    criterion_bce = torch.nn.BCELoss().to(cfg.device)
    loss = criterion_bce(seg_true, seg_pred)
    loss = torch.mean(loss)
    return loss
