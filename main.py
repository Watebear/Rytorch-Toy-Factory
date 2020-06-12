import os
import time
import datetime
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler

from data import datasets
from backbone import LeNet
from loss.angular_penalty_loss import AngularPenaltySMLoss
from loss.center_loss import CenterLoss
from torch.nn import CrossEntropyLoss
from utils.training_func import train_center_loss, train_single_loss, test


def main():
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()

    print("Implement {} on {}".format(args.model_name, args.dataset))
    if use_gpu:
        print("Currently using GPU: {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

    # dataset
    dataset = datasets.create(name=args.dataset, batch_size=args.batch_size, use_gpu=use_gpu, num_workers=args.workers)
    trainloader, testloader = dataset.trainloader, dataset.testloader

    # bachbone
    backbone = LeNet.create(name=args.model_name, num_classes=10, feat_dim=2)
    if use_gpu:
        backbone = nn.DataParallel(backbone).cuda()

    # losses and optimizers
    if args.model_name == 'center-loss':
        criterion_xent = CrossEntropyLoss()
        criterion_cent = CenterLoss(num_classes=args.num_classes, feat_dim=args.feat_dim, use_gpu=use_gpu)
        optimizer_centloss = torch.optim.SGD(criterion_cent.parameters(), lr=args.lr_cent)
    elif args.model_name == 'softmax' or 'l2_softmax':
        criterion_xent = CrossEntropyLoss()
    elif args.model_name == 'sphereface' or 'cosface' or 'arcface' :
        criterion_xent = AngularPenaltySMLoss(loss_type=args.model_name, num_classes=args.num_classes,
                                              feat_dim=args.feat_dim, use_gpu=True, s=args.s, m=args.m)
    optimizer_model = torch.optim.SGD(backbone.parameters(), lr=args.lr_model, weight_decay=5e-04, momentum=0.9)

    # lr scheduler
    if args.stepsize > 0:
        scheduler = lr_scheduler.StepLR(optimizer_model, step_size=args.stepsize, gamma=args.gamma)

    # folder to save ckpts
    model_folder = os.path.join(args.ckpt_folder, args.model_name)
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)

    # start training
    start_time = time.time()
    for epoch in range(args.max_epoch):
        print("==> Epoch {}/{}".format(epoch + 1, args.max_epoch))
        if args.model_name == 'cosface':
            train_center_loss(model=backbone, criterion_xent=criterion_xent, criterion_cent=criterion_cent,
                              optimizer_model=optimizer_model, optimizer_centloss=optimizer_centloss,
                              trainloader=trainloader, use_gpu=use_gpu, args=args)
        else:
            train_single_loss(model=backbone, criterion_xent=criterion_xent, optimizer_model=optimizer_model,
                              trainloader=trainloader, use_gpu=use_gpu, args=args)

        if args.stepsize > 0: scheduler.step()

        if args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0 or (epoch + 1) == args.max_epoch:
            print("==> Test")
            acc, err = test(backbone, testloader, use_gpu=True)
            print("Accuracy (%): {}\t Error rate (%): {}".format(acc, err))

            # save model
            if acc >= 0.90 and (epoch + 1) % args.save_freq:
                filename = "{}-{}-epoch:{}-acc:{}".format(args.dataset, args.model_name, epoch, acc)
                torch.save(backbone, os.path.join(model_folder, filename))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

    # save final model
    filename = "{}-{}-epoch:{}-acc:{}".format(args.dataset, args.model_name, epoch, acc)
    torch.save(backbone, os.path.join(model_folder, filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Welcome to Loss-Toy-Factory")
    # misc
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print_freq', type=int, default=50, help="in a epoch")
    parser.add_argument('--ckpt_folder', type=str, default='./ckpt', help="folder to save ckpts")
    parser.add_argument('--eval_freq', type=int, default=5, help="test interval")
    parser.add_argument('--save_freq', type=int, default=10, help="save model params ")
    # dataset
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashionmnist'])
    parser.add_argument('--workers', default=4, type=int, help="number of data loading workers (default: 4)")
    # model
    parser.add_argument('--model_name', type=str, default='sphereface',
                        choices=['softmax', 'l2-softmax', 'center-loss', 'sphereface', 'cosface', 'arcface'])
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--feat_dim', type=int, default=2, choices=[2, 3])
    parser.add_argument('--s', type=float, default=None, help="s for angular penalty loss")
    parser.add_argument('--m', type=float, default=None, help="m for angular penalty loss")
    parser.add_argument('--weight-cent', type=float, default=1, help="weight for center loss")
    # optimization
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_epoch', type=int, default=100, help='total epochs for training')
    parser.add_argument('--lr-model', type=int, default=0.001, help='init lr for model')
    parser.add_argument('--lr-cent', type=float, default=0.5, help="learning rate for center loss")
    parser.add_argument('--stepsize', type=int, default=20, help='lr decay per stepsize epoch')
    parser.add_argument('--gamma', type=float, default=0.5, help="learning rate decay")

    args = parser.parse_args()

    main()
