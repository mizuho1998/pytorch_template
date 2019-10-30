"""
python main.py --root_path /path/to/data/root --video_path jpg \
               --annotation_path annotation.json --label_path label.csv \
               --log_path result/log \
               --checkpoint 10 --epochs 200 --batch_size 8 --device 0


/path/to/data/root
.
├── annotaiton.json
├── label.csv
├── result
│   └── log
├── video
└── jpg

"""

import os
import torch
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from opt import parse_opts
from model.model import Net
from dataset.dataset import Dataset
from train import train_epoch
from validation import val_epoch


if __name__ == '__main__':
    opt = parse_opts()
    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        opt.labels_path = os.path.join(opt.root_path, opt.label_path)
        opt.log_path          = os.path.join(opt.root_path, opt.log_path)
        opt.log_model_path    = os.path.join(opt.log_path, 'model')
        opt.log_learning_path = os.path.join(opt.log_path, 'log')

        if not os.path.exists(opt.log_model_path):
            os.makedirs(opt.log_model_path)

        train_data_path = os.path.join(opt.video_path, 'train')
        validation_data_path = os.path.join(opt.video_path, 'validation')

    print('video_path:           ', opt.video_path)
    print('train_data_path:      ', train_data_path)
    print('validation_data_path: ', validation_data_path)
    print('annotation_path:      ', opt.annotation_path)
    print('log_path:             ', opt.log_path)
    print('log_model_path:       ', opt.log_model_path)
    print('log_learning_path:    ', opt.log_learning_path)
    print('pretrained_model_path:', opt.pretrained_model_path)

    train_dataset = Dataset(train_data_path, opt.annotation_path, opt.labels_path)
    train_loader  = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=opt.batch_size,
                                                shuffle=False,
                                                num_workers=16,
                                                pin_memory=True)

    validation_dataset = Dataset(validation_data_path, opt.annotation_path, opt.labels_path)
    validation_loader  = torch.utils.data.DataLoader(validation_dataset,
                                                batch_size=opt.batch_size,
                                                shuffle=False,
                                                num_workers=16)

    writer_train = SummaryWriter(log_dir=os.path.join(opt.log_learning_path, 'train'))
    writer_val   = SummaryWriter(log_dir=os.path.join(opt.log_learning_path, 'val'))

    model = Net()

    device = opt.device
    if opt.is_parallel:
        opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            print("multi GPUs: ", torch.cuda.device_count())
    elif device != "":
        opt.device = "cuda:" + str(device)
    else:
        opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if opt.pretrained_model_path != '':
        print("load from ", opt.pretrained_model_path)
        pretrain_log = torch.load(opt.pretrained_model_path, map_location=opt.device)
        model.load_state_dict(pretrain_log['state_dict'])
        print("pre-tran states are loaded")

    model.to(opt.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    print("trian")
    for epoch in tqdm(range(1, opt.epochs + 1)):

        train_epoch(epoch, train_loader, model, criterion, optimizer, opt, writer_train)

        val_loss, val_acc_top_1, val_acc_top_5 = val_epoch(epoch, validation_loader, model, criterion, opt)
        writer_val.add_scalar("loss/loss", val_loss, epoch * len(train_loader))
        writer_val.add_scalar("acc/acc top1", val_acc_top_1, epoch * len(train_loader))
        writer_val.add_scalar("acc/acc top5", val_acc_top_5, epoch * len(train_loader))

    writer_train.close()
    writer_val.close()

    print('Finished Training')
