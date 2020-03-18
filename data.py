import numpy as np
import scipy.sparse as sp
import torch
import pickle


def load_data(path="data/", name_list=['w', '1h', '1c', '2', '3', '6', '7', 'T', 'sI'], n_conf=10):
    print('Loading ice dataset...')

    adj_train_list = []
    feature_train_list = []
    label_train_list = []

    adj_val_list = []
    feature_val_list = []
    label_val_list = []

    adj_test_list = []
    feature_test_list = []
    label_test_list = []

    train_size = int(n_conf*.8)
    val_size = int(n_conf*.1)
    test_size = n_conf - train_size - val_size

    if train_size<1 or val_size<1 or test_size<1:
        print("Dataset is too small to split train-validation-test set!")
        exit(1)

    idx = np.linspace(0, n_conf-1, n_conf)
    np.random.shuffle(idx)
    idx_train = idx[:train_size]
    idx_val = idx[train_size:train_size+val_size]
    idx_test = idx[train_size+val_size:]

    n_class = len(name_list)

    for i, name in enumerate(name_list):
        data_fname = path + "data_{}_10frame.npz".format(name)
        adj_fname = path + "a_hat_{}_10frame.pickle".format(name)

        data_file = np.load(data_fname)

        feat_list = data_file['feat']
        n_fea = np.shape(feat_list)[2]

        with open(adj_fname, 'rb') as f:
            adj_sp_list = pickle.load(f)

        n_node = np.shape(feat_list)[1]

        feat_list = feat_list[:n_conf]
        adj_sp_list = adj_sp_list[:n_conf]

        for j, (feat, adj_sp) in enumerate(zip(feat_list, adj_sp_list)):

            feature = feat

            # normalize
            for k in range(5):
                feature[:, k] /= 5.
            feature[:, 12] /= 65.
            feature[:, 13] /= 3.5

            label = np.full(n_node, i)

            feature = torch.FloatTensor(feature)
            label = torch.LongTensor(label)
            adj = adj_sp.todense()
            adj = torch.FloatTensor(np.array(adj))

            if j in idx_train:
                adj_train_list.append(adj)
                feature_train_list.append(feature)
                label_train_list.append(label)

            elif j in idx_val:
                adj_val_list.append(adj)
                feature_val_list.append(feature)
                label_val_list.append(label)

            elif j in idx_test:
                adj_test_list.append(adj)
                feature_test_list.append(feature)
                label_test_list.append(label)

    return adj_train_list, adj_val_list, adj_test_list, feature_train_list, feature_val_list, feature_test_list, label_train_list, label_val_list, label_test_list, n_fea, n_class



def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def output_to_label(output):
    preds = output.max(1)[1]
    preds = preds.detach().numpy()
    return preds

def prepare_svm_data(features_data, labels_data):
    x = []
    y = []
    for feature in features_data:
        feature = feature.numpy()
        for f in feature:
            x.append(f)
    for label in labels_data:
        label = label.numpy()
        for l in label:
            y.append(l)
    return x, y

