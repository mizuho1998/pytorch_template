"""
python main.py --root_path /path/to/data/root --video_path jpg \
               --annotation_path annotation.json --label_path label.csv \
               --log_path result/log --pretrained_model_path result/log/model/log.pth\
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

from model import C3D
from dataset import Dataset
from opt import parse_opts
from test import test
from utils import Logger


if __name__ == '__main__':
    opt = parse_opts()
    if opt.root_path != '':
        opt.video_path      = os.path.join(opt.root_path, opt.video_path)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        opt.labels_path     = os.path.join(opt.root_path, opt.label_path)
        opt.log_path        = os.path.join(opt.root_path, opt.log_path)
        opt.log_test_path    = os.path.join(opt.log_path, 'test')

        test_data_path = os.path.join(opt.video_path, 'test')
    
    print('video_path:           ', opt.video_path)
    print('test_data_path:       ', test_data_path)
    print('annotation_path:      ', opt.annotation_path)
    print('log_path:             ', opt.log_path)
    print('pretrained_model_path:', opt.pretrained_model_path)


    test_dataset = Dataset(test_data_path, opt.annotation_path, opt.labels_path)
    test_loader  = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=opt.batch_size,
                                                shuffle=False,
                                                num_workers=16,
                                                pin_memory=True)
    
    model = C3D()

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

    print("start test")
    logger = Logger(os.path.join(opt.log_path, opt.log_test_path),
            ['outputs', 'loss', 'acc_top_1', 'acc_top_5'])

    test(test_loader, model, criterion, opt, logger)
    

