import argparse
import tqdm
import time
from torch import optim
from utils import *
from torch.utils.data import DataLoader
from SiameseNetwork import SiameseNetwork, ContrastiveLoss
import torch

# Optimization Options
parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=2, metavar='N', help='Input batch size for training (default: 20)')
parser.add_argument('--n-filters', type=int, default=128, metavar='N', help='Number of filters (default: 128)')
parser.add_argument('--hidden-size', type=int, default=128, metavar='N', help='Size of hidden layer (default: 128)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Enables CUDA training')
parser.add_argument('--epochs', type=int, default=100, metavar='N', help='Number of epochs to train (default: 360)')
parser.add_argument('--log-interval', type=int, default=500, metavar='N', help='How many batches to wait before logging training status')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='Initial learning rate (default: 1e-4)')
parser.add_argument('--lr-decay', type=float, default=0.1, metavar='LR-DECAY', help='Learning rate decay factor (default: 0.6)')
parser.add_argument('--load-pre-model', type=bool, default=False, help='load pre model')
parser.add_argument('--checkpoint-dir', default='./checkpoint/', help='path to latest checkpoint')
parser.add_argument('--schedule', type=list, default=[0.1, 0.9], metavar='S', help='Percentage of epochs to start the learning rate decay [0, 1] (default: [0.1, 0.9])')

def main():

    global args
    args = parser.parse_args()

    # Check if CUDA is enabled
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # 加载数据
    # 定义文件dataset
    train_X, test_X = csv_read()

    siamese_dataset = SiameseNetworkDataset(train_X)
    siamese_dataset_test = SiameseNetworkDataset(test_X)
    # 定义图像dataloader
    train_dataloader = DataLoader(siamese_dataset, shuffle=True, batch_size=args.batch_size)
    test_dataloader = DataLoader(siamese_dataset_test, shuffle=True, batch_size=args.batch_size)

    # 网络
    model = SiameseNetwork()
    if args.load_pre_model:
        model.load_state_dict(torch.load(args.checkpoint_dir))

    criterion = ContrastiveLoss()  # 定义损失函数
    optimizer = optim.Adam(model.parameters(), lr=args.lr)  # 定义优化器
    if args.cuda:
        print('\t* Cuda')
        model = model.cuda()
        criterion = criterion.cuda()
        lr = args.lr
        lr_step = (args.lr - args.lr * args.lr_decay) / (
                    args.epochs * args.schedule[1] - args.epochs * args.schedule[0])

    for epoch in range(args.epochs):
        # 动态lr
        if epoch > args.epochs * args.schedule[0] and epoch < args.epochs * args.schedule[1]:
            lr -= lr_step
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        train(epoch, train_dataloader, model, criterion, optimizer)
        validate(test_dataloader, model, criterion)

def train(epoch, train_dataloader, model, criterion, optimizer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_accuracy = AverageMeter()
    model.train()

    end = time.time()  # 当前时间
    for i, (image1, image2, target) in enumerate(train_dataloader):
        if args.cuda:
            image1, image2, target = image1.cuda(), image2.cuda(), target.cuda()
        # 计算加载数据时间
        data_time.update(time.time()-end)
        optimizer.zero_grad()
        output = model(image1, image2)
        train_loss = criterion(output, target)
        # 记录train_loss
        losses.update(train_loss.item(), image1.shape[0])

        # 记录训练数据的SROCC
        predicted = output
        avg_accuracy.update(computeSpearman(target, predicted)[1], image1.shape[0])

        # 记录每个batch的时间
        batch_time.update(time.time() - end)
        end = time.time()

        train_loss.backward()
        optimizer.step()
        # 设置batch的现实细粒度
        if i % args.log_interval == 0 and i > 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Avg accuracy {acc.val:.4f} ({acc.avg:.4f})'
                  .format(epoch, i, len(train_dataloader), batch_time=batch_time,
                          data_time=data_time, loss=losses, acc=avg_accuracy))

    print('Epoch: [{0}] Avg accuracy {acc.avg:.3f}; Average Loss {loss.avg:.3f}; Avg Time x Batch {b_time.avg:.3f}'
              .format(epoch, acc=avg_accuracy, loss=losses, b_time=batch_time))


def validate(test_dataloader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_accuracy = AverageMeter()
    model.eval()

    end = time.time()  # 当前时间
    for i, (image1, image2, target) in enumerate(test_dataloader):
        if args.cuda:
            image1, image2, target = image1.cuda(), image2.cuda(), target.cuda()
        # 计算加载数据时间

        output = model(image1, image2)
        test_loss = criterion(output, target)
        # 记录train_loss
        losses.update(test_loss.data[0], image1.shape[0])

        # 记录训练数据的SROCC
        predicted = output
        avg_accuracy.update(computeSpearman(target, predicted)[1], image1.shape[0])

        # 记录每个batch的时间
        batch_time.update(time.time() - end)
        end = time.time()

        # 设置batch的现实细粒度
        if i % args.log_interval == 0 and i > 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Avg accuracy {acc.val:.4f} ({acc.avg:.4f})'
                  .format(i, len(test_dataloader), batch_time=batch_time,
                          loss=losses, acc=avg_accuracy))

    print(' Avg accuracy {acc.avg:.3f}; Average Loss {loss.avg:.3f}'
          .format(acc=avg_accuracy, loss=losses))


if __name__ == '__main__':
    main()