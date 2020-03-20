from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from models import NN, GCN
from data import load_data, accuracy, prepare_svm_data

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score



# Training settings
parser = argparse.ArgumentParser()
#parser.add_argument('--fastmode', action='store_true', default=False,
#                    help='Validate during training pass.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--n_conf', type=int, default=10,
                    help='Number of configurations of molecular snapshots')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--model', type=str, default='GCN',
                    help='Network Model: SVM, NN, GCN')
parser.add_argument('--log', type=str, default=None,
                    help='Log file containing training process')
parser.add_argument('--o', type=str, default=None,
                    help='Save model')
parser.add_argument('--checkpoint', type=str, default=None,
                    help='continue training')

args = parser.parse_args()


np.random.seed(int(time.time()))
torch.manual_seed(int(time.time()))


# Load data
adj_train, adj_val, adj_test, features_train, features_val, features_test, labels_train, labels_val, labels_test, n_feat, n_class = load_data(n_conf=args.n_conf)

# Model and optimizer
if args.model == 'NN':
    model = NN(n_feat=n_feat,
               n_hid=args.hidden,
               n_class=n_class,
               dropout=args.dropout)
elif args.model == 'GCN':
    model = GCN(n_feat=n_feat,
                n_hid=args.hidden,
                n_class=n_class,
                dropout=args.dropout)
elif args.model == 'SVM':
    # dummy process
    model = NN(n_feat, args.hidden, n_class, args.dropout)
    clf = SVC(gamma='auto', verbose=True)
else:
    raise ValueError("You choose wrong network model")

optimizer = optim.Adam(model.parameters(), lr=args.lr)

if args.checkpoint is not None:
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

n_train = len(adj_train)
n_val = len(adj_val)
n_test = len(adj_test)

def train(epoch):
    if args.model == 'SVM':
        x, y = prepare_svm_data(features_test, labels_test)
        clf.fit(x, y)
        pred = clf.predict(x)
        print(accuracy_score(pred, y))

        exit(1)


    t = time.time()

    running_loss_train = 0
    running_acc_train = 0

    for i in range(n_train):
        model.train()
        optimizer.zero_grad()
        output_train = model(features_train[i], adj_train[i])
        loss_train = F.nll_loss(output_train, labels_train[i])
        acc_train = accuracy(output_train, labels_train[i])
        loss_train.backward()
        optimizer.step()
        running_loss_train += loss_train.data
        running_acc_train += acc_train

    running_loss_val = 0
    running_acc_val = 0

    for i in range(n_val):
        model.eval()
        output_val = model(features_val[i], adj_val[i])
        loss_val = F.nll_loss(output_val, labels_val[i])
        acc_val = accuracy(output_val, labels_val[i])
        running_loss_val += loss_val.data
        running_acc_val += acc_val

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(running_loss_train/n_train),
          'acc_train: {:.4f}'.format(running_acc_train/n_train),
          'loss_val: {:.4f}'.format(running_loss_val/n_val),
          'acc_val: {:.4f}'.format(running_acc_val/n_val),
          'time: {:.4f}s'.format(time.time() - t))

    return [epoch+1, running_loss_train/n_train, running_acc_train/n_train, running_loss_val/n_val, running_acc_val/n_val]



def test():

    running_loss_test = 0
    running_tot_acc_test = 0
    
    for i in range(n_test):
        model.eval()
        output_test = model(features_test[i], adj_test[i])
        loss_test = F.nll_loss(output_test, labels_test[i])
        tot_acc_test = accuracy(output_test, labels_test[i])
        running_loss_test += loss_test
        running_tot_acc_test += tot_acc_test

        idx = labels_test[i].detach().numpy()[0]

    print("Test set results:",
          "loss= {:.4f}".format(running_loss_test/n_test),
          "Total accuracy={:.4f}".format(running_tot_acc_test/n_test))
    



# Train model
log = []
t_total = time.time()
for epoch in range(args.epochs):
    log_epoch = train(epoch)
    log.append(log_epoch)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Test model
test()



# Save File
if args.log is not None:
    np.savetxt(args.log, log)

if args.o is not None:
    torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
               }, args.o)
