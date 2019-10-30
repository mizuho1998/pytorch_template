import os
from tqdm import tqdm
import torch

from utils import AverageMeter, accuracy


def train_epoch(epoch, data_loader, model, criterion, optimizer, opt, writer):

    model.train()

    losses = AverageMeter()
    accuracies_top_1 = AverageMeter()
    accuracies_top_5 = AverageMeter()

    for i, data in enumerate(tqdm(data_loader)):
        inputs, labels = data[0].to(opt.device), data[1].to(opt.device)
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        acc_top_1, acc_top_5 = accuracy(outputs, labels)

        losses.update(loss.item(), inputs.size(0))
        accuracies_top_1.update(acc_top_1, inputs.size(0))
        accuracies_top_5.update(acc_top_5, inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar("loss/loss", losses.val, (epoch - 1) * len(data_loader) + (i + 1))
        writer.add_scalar("acc/acc top1", accuracies_top_1.val, (epoch - 1) * len(data_loader) + (i + 1))
        writer.add_scalar("acc/acc top5", accuracies_top_5.val, (epoch - 1) * len(data_loader) + (i + 1))

    if epoch % opt.checkpoint == 0:
        save_file_path = os.path.join(opt.log_model_path,
                                    'save_{}.pth'.format(epoch))

        if opt.is_parallel:
            state = model.module.state_dict()
        else:
            state = model.state_dict()

        states = {
            'epoch': epoch + 1,
            'state_dict': state,
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_file_path)
