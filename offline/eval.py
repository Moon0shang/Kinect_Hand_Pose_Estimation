import os
import time
import random
import logging
import numpy as np
from tqdm import tqdm
import scipy.io as sio

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable

from dataset_apply import HandPointDataset
from network import PointNet_Plus
from utils import group_points
from parsers import init_parser


def main(opt):

    torch.cuda.set_device(opt.main_gpu)

    opt.manualSeed = 1
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    save_dir = './offline/results'

    try:
        os.makedirs(save_dir)
    except OSError:
        pass

    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S',
                        filename=os.path.join(save_dir, 'train.log'), level=logging.INFO)
    logging.info('======================================================')

    # 1. Load data
    test_data = HandPointDataset(opt=opt, train=False)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batchSize,
                                                  shuffle=False, num_workers=int(opt.workers), pin_memory=False)

    # print('#Train data:', len(train_data), '#Test data:', len(test_data))
    # print(opt)

    # 2. Define model, loss and optimizer
    netR = PointNet_Plus(opt)
    if opt.ngpu > 1:
        netR.netR_1 = torch.nn.DataParallel(netR.netR_1, range(opt.ngpu))
        netR.netR_2 = torch.nn.DataParallel(netR.netR_2, range(opt.ngpu))
        netR.netR_3 = torch.nn.DataParallel(netR.netR_3, range(opt.ngpu))

    # NOTE
    netR.load_state_dict(torch.load('./netR_58.pth'))

    netR.cuda()

    optimizer = optim.Adam(
        netR.parameters(), lr=opt.learning_rate, betas=(0.5, 0.999), eps=1e-06)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    # 3. Training and testing
    for epoch in range(opt.nepoch):
        scheduler.step(epoch)
        print('======>>>>> Online epoch: #%d, lr=%f, Test: %s <<<<<======' %
              (epoch, scheduler.get_lr()[0], "TTT"))

        test_data_length = len(test_data)
        test_mse = application(netR,  optimizer,
                               test_dataloader, test_data_length, epoch)

        # log
        logging.info('Epoch#%d: test error=%e, lr = %f' % (
            epoch, test_mse, scheduler.get_lr()[0]))


def application(netR,  optimizer, data_loader, data_length, epoch):

    # 3.1 switch to training mode
    torch.cuda.synchronize()

    netR.eval()

    out_mse = 0.0
    timer = time.time()
    for i, data in enumerate(tqdm(data_loader, 0)):
        if len(data[0]) == 1:
            continue
        torch.cuda.synchronize()
        # 3.1.1 load inputs and ground truth
        points = data
        points = points.cuda()

        inputs_level1, inputs_level1_center = group_points(points, opt)
        inputs_level1 = Variable(inputs_level1, requires_grad=False)
        inputs_level1_center = Variable(
            inputs_level1_center, requires_grad=False)

        # 3.1.2 compute output
        optimizer.zero_grad()
        estimation = netR(inputs_level1, inputs_level1_center)

        torch.cuda.synchronize()

        # 3.1.4 update training error
        outputs = estimation.data

        np.save('./offline/results/out%d.npy' % epoch, outputs.cpu())

    # time taken
    torch.cuda.synchronize()
    timer = time.time() - timer
    timer = timer / data_length
    print('==> time to learn 1 sample = %f (ms)' % (timer*1000))

    # print mse
    out_mse = out_mse / data_length
    print('mean-square error of 1 sample: %f, #train_data = %d' %
          (out_mse, data_length))

    return out_mse


if __name__ == "__main__":
    opt = init_parser()
    main(opt)
