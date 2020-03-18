from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from models import NN, GCN
from data import load_data, accuracy, output_to_label, prepare_svm_data


# Training settings
parser = argparse.ArgumentParser()
#parser.add_argument('--fastmode', action='store_true', default=False,
#                    help='Validate during training pass.')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--n_conf', type=int, default=10,
                    help='Number of snapshots used.')
parser.add_argument('--model', type=str, default='GCN',
                    help='Network Model: SVM, NN, GCN')
parser.add_argument('--checkpoint', type=str, default=None,
                    help='continue training')

args = parser.parse_args()


np.random.seed(int(time.time()))
torch.manual_seed(int(time.time()))


# Load data
_, _, adj_test, _, _, features_test, _, _, labels_test, n_feat, n_class = load_data(n_conf=args.n_conf)

# Model and optimizer
if args.model == 'NN' or args.model == "SVM":
    model = NN(n_feat=n_feat,
               n_hid=args.hidden,
               n_class=n_class,
               dropout=args.dropout)
elif args.model == 'GCN':
    model = GCN(n_feat=n_feat,
                n_hid=args.hidden,
                n_class=n_class,
                dropout=args.dropout)
else:
    raise ValueError("You choose wrong network model")

optimizer = optim.Adam(model.parameters())

if args.checkpoint is not None:
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

n_test = len(adj_test)

clf = SVC(gamma='auto', verbose=True)

def test():
    if args.model == 'SVM':
        x, y = prepare_svm_data(features_test, labels_test)
        clf.fit(x, y)
        pred = clf.predict(x)
        print(accuracy_score(pred, y))

        return 0

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
    

def test_each_model(name):
    _, _, adj_test, _, _, features_test, _, _, labels_test, _, _ = load_data(n_conf=args.n_conf, name_list=[name])
    n_test = len(adj_test)

    prob_list = np.zeros((n_test, n_class))

    if args.model == 'SVM':
        x, y = prepare_svm_data(features_test, labels_test)
        #clf = SVC(gamma='auto', verbose=False)
        #clf.fit(x, y)
        pred = clf.predict(x)

        prob_list = np.zeros(n_class)
        for p in pred:
            prob_list[p] += 1
        prob_list /= len(pred)

        print(prob_list)
        return 0

    for i in range(n_test):
        model.eval()
        output_test = model(features_test[i], adj_test[i])
        preds_test = output_to_label(output_test)

        for p in preds_test:
            prob_list[i, p] += 1
        prob_list[i] /= len(preds_test)
   
    print(np.mean(prob_list, axis=0))




# Test
test()

'''
name_list = ['w', '1h', '1c', '3', '5', '6']
for name in name_list:
    test_each_model(name)
'''
