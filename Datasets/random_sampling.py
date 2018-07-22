from scipy.io import loadmat, savemat
import scipy.sparse as sp
import numpy as np
import os
import networkx as nx

def get_stats(adjmat, labels, features, path):
    node_count = adjmat.shape[0]
    nodes = range(node_count)

    labs_dist = np.sum(labels, axis = 0)
    labs_dist = labs_dist*1.0/node_count

    multilabel = False
    if np.sum(labels) > node_count:
        multilabel = True
    elif np.sum(labels) < node_count and labels.shape[1]>1:
        raise ValueError('Some nodes have no labels!')

    symmetric = (adjmat!=adjmat.T).nnz==0   # more efficient to compare negation
    if symmetric:
        edges_count = len(adjmat.data)/2
    else:
        G = nx.from_scipy_sparse_matrix(adjmat)
        edges_count = G.number_of_edges()

    degrees = np.array(sp.csr_matrix.sum(adjmat, axis=0))
    max_degree, min_degree, mean_degree, total_degree  = np.max(degrees), np.min(degrees), np.mean(degrees), np.sum(degrees)

    singleton = np.sum(degrees == 0)

    density = edges_count*2/(node_count*(node_count-1))

    stats =  {'NODES: ': node_count, 'EDGES: ':edges_count, 'FEATURES: ':features.shape[1], 'MULTI-LABEL: ':multilabel, 'SYMMETRIC: ':symmetric,
            'DEGREES: ':total_degree, 'MEAN_DEGREE: ':mean_degree, 'MAX_DEGREE: ':max_degree, 'MIN_DEGREE: ':min_degree, 'MEAN_DEGREE: ':mean_degree,
            'DENSITY: ':density, 'SINGLETON: ':singleton, 'LABELS_DIST: \n':labs_dist}

    with open(path+'stats.txt', 'w') as f:
        for k, v in stats.items():
            f.write(k)
            f.write(str(v))
            f.write('\n')
            print(k, v)

def check_zero(labels, nodes):
    labs = labels[nodes]
    tot_labs = np.sum(labs, axis=0)

    # Some real-life datasets have missing labels or samples
    #
    # X represents presence of sample for each label in selected
    # Y represents presence of sample for each label overall
    # If Y doesn't have a sample, then it is ok for X to not have a sample also
    # Therefore condition = X'Y' + XY (Both of them don't have or both of them have)
    x = tot_labs.astype(bool)
    y = np.sum(labels, axis=0).astype(bool)
    check = np.logical_or(np.logical_and(np.logical_not(x), np.logical_not(y)),
                          np.logical_and(x, y))
    if check.all():
        return False
    else:
        print('Labels Missing: \n', tot_labs)
        return True


def get_samples(labels, per, path, count=5):
    # IMP: Ensure reproducibility
    # Set seed each time this fn is called
    np.random.seed(1234)

    total_nodes = labels.shape[0]
    test_count  = int(0.2*total_nodes)
    val_count   = int(0.2*per*total_nodes)
    train_count = int(0.8*per*total_nodes)

    for i in range(count):

        flag=True
        while flag:
            # Create new test-val-train sets for each fold
            all_nodes = np.random.permutation(total_nodes)
            test_nodes, remaining = all_nodes[:test_count], all_nodes[test_count:]
            train_nodes =  remaining[:train_count]

            # Make sure that there is no label without even a single sample in training set
            flag = check_zero(labels, train_nodes)

        val_nodes = remaining[-val_count:]

        dir = '%s/labels_random/%d/%d/'%(path, int(per*100), i+1)
        if not os.path.exists(dir):
            os.makedirs(dir)

        train = np.zeros(total_nodes, dtype=bool)
        val   = np.zeros(total_nodes, dtype=bool)
        test  = np.zeros(total_nodes, dtype=bool)

        train[train_nodes] = True
        val[val_nodes]  = True
        test[test_nodes]  = True

        print(dir)
        # print(train_nodes, val_nodes, test_nodes)
        np.save(dir+'train_ids.npy', train)
        np.save(dir+'val_ids.npy', val)
        np.save(dir+'test_ids.npy', test)



root = './'#'Datasets/'
datasets = ['ppi_gs', 'reddit', 'ppi_gs_trans', 'pubmed',
            'reddit_trans','cora', 'citeseer', 'cora_multi',
            'blogcatalog', 'facebook', 'movielens', 'amazon', 'mlgene', 'genes_fn']
percents = [0.1]
folds = 5

for data in datasets:
    path = root+data+'/'
    labels = np.load(path+'labels.npy')
    adjmat = loadmat(path+'adjmat.mat')['adjmat']
    features = np.load(path+'features.npy')
    get_stats(adjmat, labels, features, path)
    for per in percents:
        get_samples(labels, per, path, count=folds)


