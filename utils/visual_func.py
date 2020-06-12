import torch
import os
import argparse
import torch.backends.cudnn as cudnn
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from data import datasets

def visualize_2d(model, dataloader, use_gpu, prefix='testset'):
    all_features, all_labels = [], []
    correct, total = 0, 0

    with torch.no_grad():
        for data, labels in dataloader:
            if use_gpu:
                data, labels = data.cuda(), labels.cuda()
            features, outputs = model(data)
            predictions = outputs.data.max(1)[1]

            total += labels.size(0)
            correct += (predictions == labels.data).sum()

            if use_gpu:
                all_features.append(features.data.cpu().numpy())
                all_labels.append(labels.data.cpu().numpy())
            else:
                all_features.append(features.data.numpy())
                all_labels.append(labels.data.numpy())

    # accuracy
    acc = correct * 100. / total
    err = 100. - acc

    # features
    all_features = np.concatenate(all_features, 0)  # (n_batch, batch_size, feat_dim) -> ( n_batch * batch_size, feat_dim)
    all_labels = np.concatenate(all_labels, 0)

    # plot and save
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    for label_idx in range(args.num_classes):
        plt.scatter(
            all_features[all_labels == label_idx, 0],
            all_features[all_labels == label_idx, 1],
            c=colors[label_idx],
            s=1,
        )
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')


    save_folder = os.path.join(args.save_dir, args.model_name)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    filename = "{}-{}-{}-acc:{}-err:{}".format(args.model_name, args.dataset, prefix, acc, err)
    plt.title(filename)

    save_name = os.path.join(save_folder, filename)
    plt.savefig(save_name, bbox_inches='tight')
    print("Visualied results saved to {}".format(save_name))

    plt.close()


def visualize_3d(model, dataloader, use_gpu, prefix='testset'):
    all_features, all_labels = [], []
    correct, total = 0, 0

    with torch.no_grad():
        for data, labels in dataloader:
            if use_gpu:
                data, labels = data.cuda(), labels.cuda()
            features, outputs = model(data)
            predictions = outputs.data.max(1)[1]

            total += labels.size(0)
            correct += (predictions == labels.data).sum()

            if use_gpu:
                all_features.append(features.data.cpu().numpy())
                all_labels.append(labels.data.cpu().numpy())
            else:
                all_features.append(features.data.numpy())
                all_labels.append(labels.data.numpy())

    # accuracy
    acc = correct * 100. / total
    err = 100. - acc

    # features
    all_features = np.concatenate(all_features,
                                  0)  # (n_batch, batch_size, feat_dim) -> ( n_batch * batch_size, feat_dim)
    all_labels = np.concatenate(all_labels, 0)

    # plot and save
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create a sphere
    r = 1
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0 * pi:100j]
    x = r * sin(phi) * cos(theta)
    y = r * sin(phi) * sin(theta)
    z = r * cos(phi)
    ax.plot_surface( x, y, z, rstride=1, cstride=1, color='w', alpha=0.3, linewidth=0)
    ax.scatter(all_features[:, 0], all_features[:, 1], all_features[:, 2], c=labels, s=20)

    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_aspect("equal")

    save_folder = os.path.join(args.save_dir, args.model_name)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    filename = "{}-{}-{}-acc:{}-err:{}".format(args.model_name, args.dataset, prefix, acc, err)
    plt.title(filename)

    save_name = os.path.join(save_folder, filename)
    plt.savefig(save_name, bbox_inches='tight')
    print("Visualied results saved to {}".format(save_name))


def main():
    # misc
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()

    if use_gpu:
        print("Currently using GPU: {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

    # dataset
    dataset = datasets.create(name=args.dataset, batch_size=args.batch_size, use_gpu=use_gpu, num_workers=args.workers)
    trainloader, testloader = dataset.trainloader, dataset.testloader

    # model
    print("Creating model: {}".format(args.model_name))
    model = torch.load(args.modelpath)
    if use_gpu:
        model = nn.DataParallel(model).cuda()
    model.eval()

    # infer and plot
    if args.feat_dim == '2':
        visualize_2d(model, trainloader, use_gpu, prefix='trainset')
        visualize_2d(model, testloader, use_gpu, prefix='testset')
    elif args.feat_dim == '3':
        visualize_3d(model, trainloader, use_gpu, prefix='trainset')
        visualize_3d(model, testloader, use_gpu, prefix='testset')



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Let's visualize the result!")
    # misc
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--save_dir', type=str, default='./results')
    # dataset
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashionmnist'])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--workers', default=4, type=int, help="number of data loading workers (default: 4)")
    parser.add_argument('--num_classes', default=10, type=int)
    # model
    parser.add_argument('--model_name', type=str, default='sphereface',
                        choices=['softmax', 'l2-softmax', 'center-loss', 'sphereface', 'cosface', 'arcface'])
    parser.add_argument('--feat_dim', type=int, default=2, help="2: vis_2d, 3: vis_3d")
    parser.add_argument('--modelpath', type=str, default=None, help="path to the ckptfile")
    args = parser.parse_args()

    main()