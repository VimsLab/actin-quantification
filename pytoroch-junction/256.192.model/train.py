import os
import argparse
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.datasets as datasets
import torchvision.utils as vutils

from tensorboardX import SummaryWriter


from config import cfg
from utils.logger import Logger
from utils.evaluation import accuracy, AverageMeter, final_preds
from utils.misc import save_model, adjust_learning_rate
from utils.osutils import mkdir_p, isfile, isdir, join
from utils.transforms import fliplr, flip_back
from utils.loss import get_losses
from networks import network
# from dataloader.mscocoMulti import MscocoMulti
from dataloader.mscocoMulti_double_only import MscocoMulti_double_only




def main(args):
    # import pdb; pdb.set_trace()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print (device)

    writer = SummaryWriter(cfg.tensorboard_path)
    # create checkpoint dir
    counter = 0
    if not isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # create model
    model = network.__dict__[cfg.model](
        cfg.output_shape, cfg.num_class, pretrained=True)

    model = torch.nn.DataParallel(model).to(device)
    # model = model.to(device)

    # define loss function (criterion) and optimizer
    criterion_bce = torch.nn.BCELoss().to(device)
    criterion_abs = torch.nn.L1Loss().to(device)
    # criterion_abs = offset_loss().to(device)
    # criterion1 = torch.nn.MSELoss().to(device) # for Global loss
    # criterion2 = torch.nn.MSELoss(reduce=False).to(device) # for refine loss
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg.lr,
                                 weight_decay=cfg.weight_decay)

    if args.resume:
        print(args.resume)
        checkpoint_file_resume = os.path.join(args.checkpoint, args.resume+'.pth.tar')
        if isfile(checkpoint_file_resume):
            print("=> loading checkpoint '{}'".format(checkpoint_file_resume))
            checkpoint = torch.load(checkpoint_file_resume)
            pretrained_dict = checkpoint['state_dict']
            model.load_state_dict(pretrained_dict)
            args.start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(checkpoint_file_resume, checkpoint['epoch']))
            logger = Logger(join(args.checkpoint, 'log.txt'), resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(checkpoint_file_resume))
    else:
        logger = Logger(join(args.checkpoint, 'log.txt'))
        logger.set_names(['Epoch', 'LR', 'Train Loss'])

    cudnn.benchmark = True
    print('    Total params: %.2fMB' % (sum(p.numel()
                                            for p in model.parameters())/(1024*1024)*4))

    train_loader = torch.utils.data.DataLoader(
        MscocoMulti_double_only(cfg),
        batch_size=cfg.batch_size*args.num_gpus, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    for epoch in range(args.start_epoch, args.epochs):
        lr = adjust_learning_rate(
            optimizer, epoch, cfg.lr_dec_epoch, cfg.lr_gamma)
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))

        # train for one epoch
        train_loss, counter = train(train_loader, model, [
                                    criterion_abs, criterion_bce], writer, counter, optimizer, device)
        print('train_loss: ', train_loss)

        # append logger file
        logger.append([epoch + 1, lr, train_loss])

        save_model({
            'epoch': epoch + 1,
            'info': cfg.info,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, checkpoint=args.checkpoint)

    writer.export_scalars_to_json("./test.json")
    writer.close()

    logger.close()


def train(train_loader, model, criterions, writer, counter, optimizer, device):
    criterion_abs, criterion_bce = criterions
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    # Freezing batchnorm2d
    # print("Freezing mean/var of BatchNorm2d")
    # for m in model.modules():
    #     if isinstance(m, nn.BatchNorm2d):
    #         m.eval()
    #         m.weight.requires_grad = False
    #         m.bias.requires_grad = False
    # import pdb; pdb.set_trace()



    for i, (inputs, targets, meta) in enumerate(train_loader):
        input_var = torch.autograd.Variable(inputs.to(device))
        targets = targets.type(torch.FloatTensor)
        targets = torch.autograd.Variable(targets.to(device))
        # import pdb; pdb.set_trace()
        # input_var = inputs.to(device)

        endpoints_target = targets[:, 0, :, :].to(device).unsqueeze(1)
        intersections_points_target = targets[:, 1, :, :].to(device).unsqueeze(1)
        end_points_short_offsets_target = targets[:, 2:4, :, :].to(device)
        intersection_points_short_offsets_target = targets[:, 4:6, :, :].to(device)


        ground_truth = [endpoints_target,
                        intersections_points_target,
                        end_points_short_offsets_target,
                        intersection_points_short_offsets_target,
                        ]


        with torch.enable_grad():
            optimizer.zero_grad()

            outputs = model(input_var)
            loss, loss_end_pt, loss_inter_pt, loss_short_end_pt, loss_short_inter_pt = get_losses(ground_truth, outputs)

            losses.update(loss.data.item(), inputs.size(0))
            loss = loss.to(device)

            loss.backward()
            optimizer.step()
        # import pdb; pdb.set_trace()

        ##########

        writer.add_scalar('loss', loss.data.item(), counter)
        writer.add_scalar('loss_end_pt', loss_end_pt.data.item(), counter)
        writer.add_scalar('loss_inter_pt', loss_inter_pt.data.item(), counter)
        writer.add_scalar('loss_short_end_pt', loss_short_end_pt.data.item(), counter)
        writer.add_scalar('loss_short_inter_pt', loss_short_inter_pt.data.item(), counter)

        writer.add_scalar('losses.avg', losses.avg, counter)

        counter = counter + 1
        # import pdb; pdb.set_trace()
        if(i % 50 == 0 and i != 0):
            print('iteration {} | loss: {}, avg loss: {}, '
                  .format(i, loss.data.item(), losses.avg))

    return losses.avg, counter


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CPN Training')
    parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                        help='number of data loading workers (default: 12)')
    parser.add_argument('-g', '--num_gpus', default=1, type=int, metavar='N',
                        help='number of GPU to use (default: 1)')
    parser.add_argument('--epochs', default=12, type=int, metavar='N',
                        help='number of total epochs to run (default: 32)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint')

    main(parser.parse_args())
