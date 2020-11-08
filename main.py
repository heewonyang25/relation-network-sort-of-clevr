"""""""""
Pytorch implementation of "A simple neural network module for relational reasoning
Code is based on pytorch/examples/mnist (https://github.com/pytorch/examples/tree/master/mnist)
Code is downscaled version of (https://github.com/kimhc6028/relational-networks) which is modified by ttok0s7u2n5
"""""""""
from __future__ import print_function
import argparse
import os
import pickle
import random
import numpy as np
import csv

import torch
from torch.autograd import Variable

from model import RN, CNN_MLP

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Relational-Network sort-of-CLVR Example')
parser.add_argument('--model', type=str, choices=['RN', 'CNN_MLP'], default='RN', 
                    help='resume from model stored')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')

args = parser.parse_args()
torch.manual_seed(1)

if args.model=='CNN_MLP': 
  model = CNN_MLP(args)
else:
  model = RN(args)
  
model_dirs = './model'
bs = 64 # batch size
input_img = torch.FloatTensor(bs, 3, 75, 75)
input_qst = torch.FloatTensor(bs, 18)
label = torch.LongTensor(bs)

input_img = Variable(input_img)
input_qst = Variable(input_qst)
label = Variable(label)

def tensor_data(data, i):
    img = torch.from_numpy(np.asarray(data[0][bs*i:bs*(i+1)]))
    qst = torch.from_numpy(np.asarray(data[1][bs*i:bs*(i+1)]))
    ans = torch.from_numpy(np.asarray(data[2][bs*i:bs*(i+1)]))

    input_img.data.resize_(img.size()).copy_(img)
    input_qst.data.resize_(qst.size()).copy_(qst)
    label.data.resize_(ans.size()).copy_(ans)


def cvt_data_axis(data):
    img = [e[0] for e in data]
    qst = [e[1] for e in data]
    ans = [e[2] for e in data]
    return (img,qst,ans)

    
def train(epoch, rel, norel):
    model.train()

    if not len(rel[0]) == len(norel[0]):
        print('Not equal length for relation dataset and non-relation dataset.')
        return
    
    random.shuffle(rel)
    random.shuffle(norel)

    rel = cvt_data_axis(rel)
    norel = cvt_data_axis(norel)

    acc_rels = []
    acc_norels = []

    l_binary = []
    l_unary = []

    for batch_idx in range(len(rel[0]) // bs):

        tensor_data(rel, batch_idx)
        accuracy_rel, loss_binary = model.train_(input_img, input_qst, label)
        acc_rels.append(accuracy_rel.item())
        l_binary.append(loss_binary.item())

        tensor_data(norel, batch_idx)
        accuracy_norel, loss_unary = model.train_(input_img, input_qst, label)
        acc_norels.append(accuracy_norel.item())
        l_unary.append(loss_unary.item())

        if batch_idx % 10 == 0: # 10 batches to wait before logging training status
            print('Train Epoch: {} [{}/{} ({:.0f}%)] '
                  'Relations accuracy: {:.0f}% | Non-relations accuracy: {:.0f}%'.format(
                   epoch,
                   batch_idx * bs * 2,
                   len(rel[0]) * 2,
                   100. * batch_idx * bs / len(rel[0]),
                   accuracy_rel,
                   accuracy_norel))

    avg_acc_binary = sum(acc_rels) / len(acc_rels)
    avg_acc_unary = sum(acc_norels) / len(acc_norels)

    avg_loss_binary = sum(l_binary) / len(l_binary)
    avg_loss_unary = sum(l_unary) / len(l_unary)

    # return average accuracy
    return avg_acc_binary, avg_acc_unary

def test(epoch, rel, norel):
    model.eval()
    if not len(rel[0]) == len(norel[0]):
        print('Not equal length for relation dataset and non-relation dataset.')
        return
    
    rel = cvt_data_axis(rel)
    norel = cvt_data_axis(norel)

    accuracy_rels = []
    accuracy_norels = []

    loss_binary = []
    loss_unary = []

    for batch_idx in range(len(rel[0]) // bs):
        tensor_data(rel, batch_idx)
        acc_bin, l_bin = model.test_(input_img, input_qst, label)
        accuracy_rels.append(acc_bin.item())
        loss_binary.append(l_bin.item())

        tensor_data(norel, batch_idx)
        acc_un, l_un = model.test_(input_img, input_qst, label)
        accuracy_norels.append(acc_un.item())
        loss_unary.append(l_un.item())

    accuracy_rel = sum(accuracy_rels) / len(accuracy_rels)
    accuracy_norel = sum(accuracy_norels) / len(accuracy_norels)
    print('\n=> Test set: Relations accuracy: {:.0f}% | Non-relations accuracy: {:.0f}%\n'.format(
        accuracy_rel, accuracy_norel))

    loss_binary = sum(loss_binary) / len(loss_binary)
    loss_unary = sum(loss_unary) / len(loss_unary)

    return accuracy_rel, accuracy_norel

    
def load_data():
    print('loading data...')
    dirs = './data'
    filename = os.path.join(dirs,'sort-of-clevr.pickle')
    with open(filename, 'rb') as f:
      train_datasets, test_datasets = pickle.load(f)
    rel_train = []
    rel_test = []
    norel_train = []
    norel_test = []
    print('processing data...')

    for img, relations, norelations in train_datasets:
        img = np.swapaxes(img, 0, 2)
        for qst,ans in zip(relations[0], relations[1]):
            rel_train.append((img,qst,ans))
        for qst,ans in zip(norelations[0], norelations[1]):
            norel_train.append((img,qst,ans))

    for img, relations, norelations in test_datasets:
        img = np.swapaxes(img, 0, 2)
        for qst,ans in zip(relations[0], relations[1]):
            rel_test.append((img,qst,ans))
        for qst,ans in zip(norelations[0], norelations[1]):
            norel_test.append((img,qst,ans))
    
    return (rel_train, rel_test, norel_train, norel_test)

rel_train, rel_test, norel_train, norel_test = load_data()

try:
    os.makedirs(model_dirs)
except:
    print('directory {} already exists'.format(model_dirs))

with open(f'./{args.model}_1_log.csv', 'w') as log_file:
    csv_writer = csv.writer(log_file, delimiter=',')
    csv_writer.writerow(['epoch', 'train_acc_rel',
                     'train_acc_norel', 'test_acc_rel', 'test_acc_norel'])

    print(f"----Training {args.model}----")

    for epoch in range(1, args.epochs + 1):
        train_acc_binary, train_acc_unary = train(
            epoch, rel_train, norel_train)
        test_acc_binary, test_acc_unary = test(
            epoch, rel_test, norel_test)

        csv_writer.writerow([epoch, train_acc_binary,
                         train_acc_unary, test_acc_binary, test_acc_unary])
        model.save_model(epoch)
