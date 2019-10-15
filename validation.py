import torch
import sys
from tqdm import tqdm

from utils import AverageMeter, accuracy


def val_epoch(epoch, data_loader, model, criterion, opt, writer):

    model.eval()

    losses = AverageMeter()
    accuracies_top_1 = AverageMeter()
    accuracies_top_5 = AverageMeter()

    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader)):
            inputs, labels = data[0].to(opt.device), data[1].to(opt.device)
            outputs = model(inputs)

            loss = criterion(outputs, labels_all)
            acc_top_1, acc_top_5 = accuracy(outputs, labels)

            losses.update(loss.item(), inputs.size(0))
            accuracies_top_1.update(acc_top_1, inputs.size(0) )
            accuracies_top_5.update(acc_top_5, inputs.size(0) )

    return losses.avg, accuracies_top_1.avg, accuracies_top_5.avg
