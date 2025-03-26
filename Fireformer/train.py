from __future__ import print_function, division
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import os
import torch.nn as nn
import torch.utils.data
import numpy as np
from tqdm import tqdm
import argparse
import logging
import sys
from torch.utils.data import DataLoader, random_split
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from utils.sequenceDataset import SeqDataSet
from utils.evalNet import eval_net
from myNet.Fireformer import Fireformer


make_val = False
seed = 7755
rank = 7


def train_net(net,
              patch_folders,
              device,
              dir_checkpoint,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True, ):
    dataset = SeqDataSet(patch_folders, aug=True)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True,
                              drop_last=False)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)

    writer = SummaryWriter(comment=f'_Name_{net.name}_Epoch_{epochs}_LR_{lr}_BS_{batch_size}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
    ''')

    optimizer = torch.optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='patch') as pbar:
            for batch in train_loader:
                # patchs = batch["patch"][:, :6, :, :]
                patchs = batch["patch"][:, :, :, :]
                labels = batch["label"]
                patchs = patchs.to(device=device, dtype=torch.float32)
                labels_type = torch.long
                labels = labels.to(device=device, dtype=labels_type)
                labels = torch.squeeze(labels, dim=1)
                labels = torch.squeeze(labels, dim=1)

                labels_pred = net(patchs)
                pred_classes = torch.max(labels_pred, dim=1)[1]
                loss = criterion(labels_pred, labels)
                # print('loss:', loss)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(patchs.shape[0])
                global_step += 1
                if global_step % int((n_train / batch_size) / 2) == 0 and make_val:  # n_train // (100 * batch_size)
                    val_score, val_acc = eval_net(net, val_loader, device, sum=n_val)
                    scheduler.step(val_score)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
                    writer.add_scalar('Acc/val', val_acc, global_step)

                    logging.info('Validation Loss: {}'.format(val_score))
                    writer.add_scalar('Loss/Val', val_score, global_step)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

        if save_cp or epoch == 0:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'/{net.name}_20250310_1200_CP_epoch{epoch + 1}.pth')
            logging.info(
                f'Checkpoint {epoch + 1} saved !' + dir_checkpoint + f'/{net.name}_20250310_1200_CP_epoch{epoch + 1}.pth')

    writer.close()
    if not save_cp:
        torch.save(net.state_dict(),
                   dir_checkpoint + f'/{net.name}_20250310_1200_CP_epoch{epochs + 0}.pth')


def get_args():
    parser = argparse.ArgumentParser(description='Train the FireNet on patches and labels.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=500,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=200,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=1e-6,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str,
                        # default="D:/temp_model/noaug/SeqNet_SCT_20220922_1640_CP_epoch300.pth",  # SeqNet_SCT_20220819_1220_CP_epoch100
                        default="",
                        help='Load model from a .pth file')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('-pf', '--patch_folder', dest='patch_folder', type=str,
                        default=[
                            "D:/DataSet/2018",
                            "D:/DataSet/2018com",
                            "D:/DataSet/2019",
                            "D:/DataSet/2019com",
                            "D:/DataSet/2020",
                            "D:/DataSet/2020com",
                        ],
                        help='The folder of patches')
    parser.add_argument('-dc', '--dir_checkpoint', dest='dir_checkpoint', type=str,
                        default="D:/temp_model/fireformer",
                        help='The folder of models')

    return parser.parse_args()


if __name__ == '__main__':
    seed = 666
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    print(torch.cuda.is_available())
    device = torch.device('cuda')
    logging.info(f'Using device {device}')

    net = Fireformer()

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)

    try:
        train_net(net=net,
                  patch_folders=args.patch_folder,
                  device=device,
                  dir_checkpoint=args.dir_checkpoint,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  val_percent=args.val / 100,
                  save_cp=True)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
