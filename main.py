<<<<<<< HEAD
from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import sklearn.metrics as metrics
from torch.utils.data import DataLoader
from util import cal_loss,save_loss
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import ModelNet40
from model import Attentive_Pooling,Mymodel,DGCNN

def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

def train(args):
    train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points), num_workers=args.nthreads,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=args.nthreads,
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if args.model == 'dgcnn':
        model = DGCNN(args).to(device)
    else:
        model = Mymodel(args).to(device)
    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    
    criterion = cal_loss

    best_test_acc = 0
    train_data={
        'loss':[],
        'average_accuracy':[],
        'weighted_accuracy':[]
    }
    test_data={
        'loss': [],
        'average_accuracy': [],
        'weighted_accuracy': []
    }
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        if args.opt_switch > 0 and epoch > args.opt_switch:
            cur_lr = opt.param_groups[0]['lr']
            opt = optim.SGD(model.parameters(), lr=cur_lr*100, momentum=args.momentum, weight_decay=1e-4)
            scheduler = CosineAnnealingLR(opt, args.epochs - args.opt_switch, eta_min=args.lr)
            
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        for data, label in train_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1) #b*n*d->b*d*n
            batch_size = data.size()[0]
            opt.zero_grad()
            logits = model(data)
            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        scheduler.step()
        final_loss = train_loss*1.0/count
        ave_accuracy = metrics.accuracy_score(train_true, train_pred)
        weighted_accuracy = metrics.balanced_accuracy_score(train_true, train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 final_loss,
                                                                                 ave_accuracy,
                                                                                 weighted_accuracy)
        train_data['loss'].append(final_loss)
        train_data['average_accuracy'].append(ave_accuracy)
        train_data['weighted_accuracy'].append(weighted_accuracy)
        print(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits = model(data)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_final_loss = test_loss*1.0/count
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                              test_final_loss,
                                                                              test_acc,
                                                                              avg_per_class_acc)
        test_data['loss'].append(test_final_loss)
        test_data['average_accuracy'].append(test_acc)
        test_data['weighted_accuracy'].append(avg_per_class_acc)
        print(outstr)

        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'checkpoints/model%s.t7' % epoch)

        save_loss(args.exp_name,
                  [train_data['loss'], train_data['average_accuracy'], train_data['weighted_accuracy']],
                  [test_data['loss'], test_data['average_accuracy'], test_data['weighted_accuracy']],
                  epoch + 1
                  )
        print("Save loss figure.")
        print("############################################################")



def test(args, io):
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points),
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    model = Mymodel(args).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    for data, label in test_loader:

        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        logits = model(data)
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    io.cprint(outstr)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Learning')
    parser.add_argument('--exp_name', type=str, default='default')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=24, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=24, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--model', type=str, default='mymodel', metavar='N',
                        choices=['mymodel', 'dgcnn'],
                        help='Model to use, [mymodel, dgcnn]')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=512, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--heads', type=int, default=4, metavar='N',
                        help='number of heads')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--nthreads', type=int, default=4, help='Num of worker for data loading')
    parser.add_argument('--opt_switch', type=int, default=0)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    _init_()

    if args.debug:
        args.no_cuda = True
        args.batch_size = 2
        args.num_points = 24
        args.emb_dims = 36
        args.nthreads = 1
    if args.opt_switch:
        args.use_sgd = False
    print('Args: \n', args)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if not args.eval:
        train(args)
    else:
        test(args)