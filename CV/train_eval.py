
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel

import torch.optim
import torch.utils.data
from sklearn.metrics import f1_score
import numpy as np


def setup_seed(seed):
    import random
    print("Random Seed: ", seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False


def train(train_loader, val_loader, test_loader,
          model, criterion, optimizer, lr_scheduler,
          epochs, device, save_dir, print_freq=50):

    test_perf = 0
    best_val_perf = 0
    final_bias = 0

    for epoch in range(epochs):
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))

        losses = AverageMeter()
        top1 = AverageMeter()
        # switch to train mode
        model.train()
        for i, (input, target) in enumerate(train_loader):
            target = target.to(device)
            input_var = input.to(device)
            target_var = target
            # compute output
            output, _ = model(input_var)
            loss = criterion(output, target_var)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            output = output.float()
            loss = loss.float()
            # measure accuracy and record loss
            acc, predictions = accuracy(output.data, target)
            prec1 = acc[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch, i, len(train_loader), loss=losses, top1=top1))

        lr_scheduler.step()
        # evaluate on validation set
        score = eval(val_loader, model, criterion, device)
        is_best = score > best_val_perf
        best_val_perf = max(score, best_val_perf)
        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_val_perf': best_val_perf,
        }, is_best, filename=os.path.join(save_dir, 'model.th'))
        if is_best:
            test_score = eval(test_loader, model, criterion, device)
            test_perf = test_score
            final_bias = abs(best_val_perf - test_score)
            print("val perf {:.3f} test perf {:.3f} bias {:.3f}".format(best_val_perf, test_perf, final_bias))

    print("best_val_perf", best_val_perf)
    print(final_bias)
    return test_perf, best_val_perf, final_bias


def eval(loader, model, criterion, device):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    all_prec, all_target = [], []
    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            target = target.to(device)
            input_var = input.to(device)
            target_var = target.to(device)

            # compute output
            output,_ = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            acc, predictions = accuracy(output.data, target)
            prec1 = acc[0]
            all_prec.append(predictions.squeeze(dim=0).cpu())
            all_target.append(target.cpu())

            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    # print(' * Prec@1 {top1.avg:.3f}'
    #       .format(top1=top1))
    all_prec = torch.cat(all_prec, dim=0).numpy()
    all_target = torch.cat(all_target, dim=0).numpy()
    f1_macro = f1_score(all_target, all_prec, average='macro')
    return f1_macro # top1.avg,


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    if is_best:
        torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res, pred