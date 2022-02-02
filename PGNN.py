import scipy.io as spio
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adagrad, Adadelta, Adam, NAdam, RMSprop, SGD
from functools import partial

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def root_mean_squared_error(y_true, y_pred):
    return (y_pred - y_true).pow(2).mean(dim=-1).sqrt()


def density(temp):
    # function for computing the density given the temperature(nx1 matrix)
    return 1000 * (1 - (temp + 288.9414) * (temp - 3.9863)**2 / (508929.2 * (temp + 68.12963)))


def phy_loss_mean(udendiff):
    # useful for cross-checking training
    return F.relu(udendiff).mean()


def combined_loss(y_true, y_pred, udendiff, lam):
    # function to calculate the combined loss = sum of rmse and phy based loss
    return root_mean_squared_error(y_true, y_pred) + lam * F.relu(udendiff).mean()


def PGNN_train_test(optimizer_name, opter, drop_frac, use_YPhy, iteration, n_layers, n_nodes, tr_size, lamda, lake_name):

    # Hyper-parameters of the training process
    batch_size = 1000
    num_epochs = 100

    # Load features (Xc) and target values (Y)
    data_dir = './datasets/'
    filename = lake_name + '.mat'
    mat = spio.loadmat(data_dir + filename, squeeze_me=True, variable_names=['Y', 'Xc_doy', 'Modeled_temp'])
    Xc = mat['Xc_doy']
    Y = mat['Y']

    trainX, trainY = Xc[:tr_size, :], Y[:tr_size]
    testX, testY = Xc[tr_size:, :], Y[tr_size:]

    # Loading unsupervised data
    unsup_filename = lake_name + '_sampled.mat'
    unsup_mat = spio.loadmat(data_dir+unsup_filename, squeeze_me=True, variable_names=['Xc_doy1', 'Xc_doy2'])

    uX1 = unsup_mat['Xc_doy1']  # Xc at depth i for every pair of consecutive depth values
    uX2 = unsup_mat['Xc_doy2']  # Xc at depth i + 1 for every pair of consecutive depth values

    if use_YPhy == 0:
        # Removing the last column from uX (corresponding to Y_PHY)
        uX1 = uX1[:, :-1]
        uX2 = uX2[:, :-1]

    # Creating the model
    model = nn.Sequential()
    for layer in np.arange(n_layers):
        if layer == 0:
            model.add_module(f'l-{layer}', nn.Linear(np.shape(trainX)[1], n_nodes))
            model.add_module(f'a-{layer}', nn.ReLU())
        else:
            model.add_module(f'l-{layer}', nn.Linear(n_nodes, n_nodes))
            model.add_module(f'a-{layer}', nn.ReLU())
        model.add_module(f'd-{layer}', nn.Dropout(p=drop_frac))

    model.add_module('l-l', nn.Linear(n_nodes, 1))
    model.to(device)
    opt = opter(model.parameters())

    # physics-based regularization
    uin1 = torch.tensor(uX1, dtype=torch.float).to(device)  # input at depth i
    uin2 = torch.tensor(uX2, dtype=torch.float).to(device)  # input at depth i + 1
    lam = torch.tensor(lamda, dtype=torch.float).to(device)  # regularization hyper-parameter

    # train set
    train_set = TensorDataset(torch.tensor(trainX, dtype=torch.float), torch.tensor(trainY, dtype=torch.float))
    train_loader = DataLoader(train_set, batch_size=batch_size)

    for e in range(num_epochs):
        for i, (b_x, b_y) in enumerate(train_loader):
            uout1 = model(uin1)  # model output at depth i
            uout2 = model(uin2)  # model output at depth i + 1
            udendiff = (density(uout1) - density(uout2))  # difference in density estimates at every pair of depth values

            pred = model(b_x.to(device)).squeeze(-1)
            opt.zero_grad()
            totloss = combined_loss(b_y.to(device), pred, udendiff, lam)
            totloss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            phyloss = phy_loss_mean(udendiff)
            print(f'epoch {e} batch {i} \t total_loss: {totloss.item():.10} \t phyloss: {phyloss.item():.10}')

    # test set
    test_set = TensorDataset(torch.tensor(testX, dtype=torch.float), torch.tensor(testY, dtype=torch.float))
    test_loader = DataLoader(test_set, batch_size=batch_size)

    totloss_sum = 0
    phyloss_sum = 0
    for b_x, b_y in test_loader:
        uout1 = model(uin1)  # model output at depth i
        uout2 = model(uin2)  # model output at depth i + 1
        udendiff = (density(uout1) - density(uout2))  # difference in density estimates at every pair of depth values
        with torch.no_grad():
            pred = model(b_x.to(device)).squeeze(-1)
            totloss = combined_loss(b_y.to(device), pred, udendiff, lam)
            phyloss = phy_loss_mean(udendiff)
            totloss_sum += totloss * b_x.shape[0]
            phyloss_sum += phyloss_sum * b_x.shape[0]

    test_totloss = totloss_sum / testX.shape[0]
    test_phyloss_sum = phyloss_sum/testX.shape[0]
    print(f'iter: {iteration} useYPhy: {use_YPhy} nL: {n_layers} nN: {n_nodes} lamda: {lamda} trsize: {tr_size} Totloss: {test_totloss} PhyLoss: {test_phyloss_sum}')


if __name__ == '__main__':
    # Main Function

    # List of optimizers to choose from
    optimizer_names = ['Adagrad', 'Adadelta', 'Adam', 'Nadam', 'RMSprop', 'SGD', 'NSGD']
    optimizer_vals = [partial(Adagrad, lr=0.001), partial(Adadelta, lr=0.001), partial(Adam, lr=0.001), partial(NAdam, lr=0.001),
                      partial(RMSprop, lr=0.001), partial(SGD, lr=0.01), partial(SGD, lr=0.01, nesterov=True)]

    # selecting the optimizer
    optimizer_num = 2  # Adam
    optimizer_name = optimizer_names[optimizer_num]
    optimizer_val = optimizer_vals[optimizer_num]

    # Selecting Other Hyper-parameters
    drop_frac = 0  # Fraction of nodes to be dropped out
    use_YPhy = 1  # Whether YPhy is used as another feature in the NN model or not
    n_layers = 2  # Number of hidden layers
    n_nodes = 12  # Number of nodes per hidden layer

    # set lamda=0 for pgnn0
    lamda = 1000*0.5  # Physics-based regularization constant

    # Iterating over different training fractions and splitting indices for train-test splits
    trsize_range = [5000, 2500, 1000, 500, 100]
    iter_range = np.arange(1)  # range of iteration numbers for random initialization of NN parameters

    # default training size = 5000
    tr_size = trsize_range[0]

    # List of lakes to choose from
    lake = ['mendota', 'mille_lacs']
    lake_num = 0  # 0 : mendota , 1 : mille_lacs
    lake_name = lake[lake_num]
    # iterating through all possible params
    for iteration in iter_range:
        PGNN_train_test(optimizer_name, optimizer_val, drop_frac, use_YPhy,
                        iteration, n_layers, n_nodes, tr_size, lamda, lake_name)
