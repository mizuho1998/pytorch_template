import torch
from tqdm import tqdm
import numpy as np
import os


def test(data_loader, model, criterion, opt):

    model.eval()

    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader)):
            inputs, labels = data[0].to(opt.device), data[1].to(opt.device)
            outputs = model(inputs)

            np.savez(os.path.join(opt.log_test_path + '.npz'),
                    outputs=outputs.cpu().numpy(),
                    labels=labels.cpu().numpy())
