import argparse
import numpy as np

import torch
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import MNIST_Seq, MNIST_SeqOnline  # choose one
from models import MNIST_Adder, MNIST_AdderCNN  # choose one
from routines import train, test
import plotter


def main():
    parser = argparse.ArgumentParser(description='Sum MNIST digits.')
    parser.add_argument('-t', '--train', default=False, metavar='', help='train the model')
    parser.add_argument('-o', '--output', default='./mnist_adder_cnn.pth', metavar='', help='output model file')
    parser.add_argument('-d', '--data_path', default='./infimnist_data', metavar='', help='data directory')
    parser.add_argument('--n_train', default=None, metavar='', help='number of training examples')
    parser.add_argument('--n_test', default=5000, metavar='', help='number of test examples')
    parser.add_argument('--n_valid', default=1000, metavar='', help='number of validation examples')
    parser.add_argument('--max_size_train', default=10, metavar='', help='maximum size of training sets')
    parser.add_argument('--min_size_test', default=5, metavar='', help='minimum size of test sets')
    parser.add_argument('--max_size_test', default=100, metavar='', help='maximum size of test sets')
    parser.add_argument('--lr', default=1e-3, metavar='', help='learning rate')
    parser.add_argument('--epochs', default=100, metavar='', help='number of training epochs')
    parser.add_argument('--batch_size', default=128, metavar='', help='batch size')
    parser.add_argument('--use_cuda', default=True, metavar='', help='use CUDA capable GPU')
    parser.add_argument('--use_visdom', default=False, metavar='', help='use Visdom to visualize plots')
    parser.add_argument('--visdom_env', default='MNIST Adder', metavar='', help='Visdom environment name')
    parser.add_argument('--visdom_port', default=8888, metavar='', help='Visdom port')
    args = vars(parser.parse_args())

    device = 'cuda:0' if (args['use_cuda'] and torch.cuda.is_available()) else 'cpu:0'

    dataset = MNIST_SeqOnline  # or MNIST_Seq
    model = MNIST_AdderCNN()  # or MNIST_Adder
    model.to(device)
    loss = F.l1_loss
    optimizer = Adam(model.parameters(), lr=args['lr'])
    print(model)

    if args['train']:
        if args['n_train']:
            train_set = dataset(pack=0,
                                num_examples=args['n_train'],
                                max_seq_len=args['max_size_train'])
        else:
            train_set = dataset(pack=0,
                                max_seq_len=args['max_size_train'])
        train_loader = DataLoader(train_set,
                                  batch_size=args['batch_size'],
                                  shuffle=True)

        if args['n_valid'] > 0:
            valid_set = dataset(pack=1,
                                num_examples=args['n_valid'],
                                max_seq_len=args['max_size_train'])
            valid_loader = DataLoader(valid_set,
                                      batch_size=args['batch_size'],
                                      shuffle=False)
        else:
            valid_loader = None

        if args['use_visdom']:
            train_plt = plotter.VisdomLinePlotter(env_name=args['visdom_env'],
                                                  port=args['visdom_port'])
        else:
            train_plt = None

        print('Train on {} samples, validate on {} samples'.format(
            len(train_set), len(valid_set)))

        train(model, loss, optimizer, args['epochs'], train_loader,
              valid_loader=valid_loader, device=device, visdom=train_plt, model_path=args['output'])

    if args['n_test'] > 0:
        model.load_state_dict(torch.load(args['output']))

        if args['use_visdom']:
            test_plt = plotter.VisdomDictPlotter(env_name=args['visdom_env'],
                                                 port=args['visdom_port'])
        else:
            test_plt = None

        test_set = dataset(pack=np.random.randint(2, 8),
                           num_examples=args['n_test'],
                           rand_seq_len=False)
        test_loader = DataLoader(test_set,
                                 batch_size=args['batch_size'],
                                 shuffle=False)

        test(model, loss, test_loader,
             size_range=[args['min_size_test'], args['max_size_test']+1],
             device=device, visdom=test_plt)

if __name__ == '__main__':
    main()
