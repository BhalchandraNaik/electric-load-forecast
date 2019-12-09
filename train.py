import numpy as np
import pandas as pd
from datetime import date, timedelta, datetime
import os
import pickle
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.autograd import Variable
from matplotlib.pyplot import figure

np.set_printoptions(suppress=True)
CUDA_ENABLED = True

if CUDA_ENABLED:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


def get_data(file_name):
    data = pd.read_csv(file_name)
    data = data.to_numpy()
    for i in range(data.shape[0]):
        data[i, 1] = datetime.strptime(data[i, 1], "%m%d%Y %H:%M").date()
    power = data[:, 2].astype(np.float64)
    indexes = np.isnan(power)
    cleaned_data = data[~indexes, :]
    return cleaned_data

def embed_data(lead_time, embed_dim, from_date, to_date, data, t_idx, dt_idx):
    start = np.where(data[:, dt_idx] == from_date)[0][0]
    end = np.where(data[:, dt_idx] == to_date)[0][-1]
    X, y = [], []
    for idx in range(start, end+1):
        xx = data[idx-(embed_dim+lead_time-1):idx-lead_time+1, 3:]
        yy = data[idx, 2]
        X.append(xx)
        y.append(yy)
    return np.array(X), np.array(y)

def get_data_between_time(data, from_date, to_date, dt_idx):
    start = np.where(data[:, dt_idx] == from_date)[0][0]
    end = np.where(data[:, dt_idx] == to_date)[0][-1]
    return data[start:(end+1), :]


def normalize(data):
    mean_out, stds_out = 0, 0
    for i in range(2, data.shape[1]):
        if i == 2:
            mean_out = np.mean(data[:, i])
            stds_out = np.std(data[:, i])
        data[:, i] = (data[:, i] - np.mean(data[:, i]))/np.std(data[:, i])
    return data, mean_out, stds_out

def denormalize(y, mean, std):
    return y*std + mean


class Autoencoder(nn.Module):
    def __init__(self, input_size, down_size):
        super().__init__()
        self.input_size = input_size
        self.down_size = down_size

        self.factor = input_size / down_size
        self.encoded_size = self.down_size

        self.factor_root = np.sqrt(self.factor)
        self.first_squash = int(input_size / self.factor_root)

        self.linear_encode1 = nn.Linear(self.input_size, self.first_squash)
        self.linear_encode2 = nn.Linear(self.first_squash, self.encoded_size)

        self.linear_decode1 = nn.Linear(self.encoded_size, self.first_squash)
        self.linear_decode2 = nn.Linear(self.first_squash, self.input_size)

        # Activation
        self.first_squash_fn = nn.ELU()
        self.encode_fn = nn.Sigmoid()

    def forward(self, xx, decode=True, no_act=False):
        if no_act:
            xx = self.linear_encode1(xx)
            xx = self.linear_encode2(xx)

            if decode:
                xx = self.linear_decode1(xx)
                xx = self.linear_decode2(xx)
        else:
            xx = self.first_squash_fn(self.linear_encode1(xx))
            xx = self.encode_fn(self.linear_encode2(xx))

            if decode:
                xx = self.encode_fn(self.linear_decode1(xx))
                xx = self.first_squash_fn(self.linear_decode2(xx))

        return xx

    def train(self, data, batch_size, epochs=30, lr=0.01, no_act=False):
        learning_rate = 0.01
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        losses = []
        # Train the model
        for epoch in range(epochs):
            indices = torch.randperm(data.size()[0])
            data = data[indices, :]
            for i in range(0, data.size()[0], batch_size):
                xx = data[i: i + batch_size, :]
                outputs = self.forward(xx, no_act=no_act)
                optimizer.zero_grad()

                loss = criterion(outputs, xx)
                losses.append(loss.item())

                loss.backward()
                optimizer.step()
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
        return losses

class QuantileRegression(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, seq_length, alpha, bidirectional=True):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.num_layers = num_layers
        self.alpha = alpha

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True, bidirectional=False)

        self.fc = nn.Linear(hidden_size, 10)
        self.fc_out = nn.Linear(10, output_size)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        batch_size = x.size(0)
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out.view(batch_size * self.num_layers, self.hidden_size)

        fc1 = self.fc(h_out)
        out = self.fc_out(fc1)

        return out

    def train_model(self, trainX, trainY, lr, epochs, sample_size, verbose=True):
        learning_rate = lr
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        losses = []
        num_epochs = epochs

        for epoch in range(num_epochs):
            indices = torch.randperm(trainX.size()[0])
            trainX, trainY = trainX[indices, :, :], trainY[indices]
            for i in range(0, trainX.size()[0], sample_size):
                xx = trainX[i: i + sample_size, :, :]
                yy = trainY[i: i + sample_size]
                outputs = self.forward(xx)
                outputs = outputs.view(self.num_layers, self.output_size, -1)
                optimizer.zero_grad()
                alpha = self.alpha
                # obtain the loss function
                loss = self.quantile_loss(outputs[-1, :, :], yy)
                losses.append(loss.item())
                loss.backward()

                optimizer.step()
            if verbose:
                print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

        return losses

    def train_sgd(self, trainX, trainY, lr, epochs, sample_size):
        learning_rate = lr
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        losses = []
        num_epochs = epochs
        for epoch in range(num_epochs):
            for xx, yy in zip(trainX, trainY):
                xx = xx.view(1, self.seq_length, self.input_size)
                outputs = self.forward(xx)
                optimizer.zero_grad()

                # obtain the loss function
                loss = self.quantile_loss(outputs, yy)
                losses.append(loss)
                loss.backward()

                optimizer.step()
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
        return losses

    def quantile_loss(self, output, target):
        covered_flag = (output <= target).float()
        uncovered_flag = (output > target).float()
        return torch.mean(
            (target - output) * (self.alpha) * covered_flag + (output - target) * (1 - self.alpha) * uncovered_flag)




if __name__ == "__main__":
    data = get_data('data/task_1/L1-train.csv')
    data, mean_out, stds_out = normalize(data)
    embedded_data, target = embed_data(
        lead_time=6,
        embed_dim=12,
        from_date=date(month=2, year=2005, day=1),
        to_date=date(month=2, year=2006, day=1),
        data=data,
        t_idx=2,
        dt_idx=1
    )

    ae_data = get_data_between_time(
        data=data,
        from_date=date(month=1, year=2005, day=1),
        to_date=date(month=12, year=2005, day=31),
        dt_idx=1
    )

    ae_data = Variable(torch.from_numpy(ae_data[:, 3:].astype(np.float32))).cuda()

    autoencoder = Autoencoder(
        input_size=25,
        down_size=15
    )

    losses = autoencoder.train(
        data=ae_data,
        batch_size=20,
        epochs=200,
        lr=0.000005,
        no_act=True
    )

    dataTensor = Variable(torch.from_numpy(data[:, 3:].astype(np.float32))).cuda()
    encoded_data_without_date = autoencoder.forward(dataTensor, decode=False, no_act=True).cpu().detach().numpy()
    encoded_data = np.concatenate((data[:, :3], encoded_data_without_date), axis=1)
    embedded_encoded_data, target = embed_data(
        lead_time=6,
        embed_dim=12,
        from_date=date(month=1, year=2006, day=1),
        to_date=date(month=3, year=2006, day=31),
        data=encoded_data,
        t_idx=2,
        dt_idx=1
    )

    train_x = embedded_data.astype(np.float32)
    train_y = target.astype(np.float32)

    lead_time = [6, 12]
    embed_dims = [12, 24, 36]
    quantiles = np.arange(0.05, 1.0, 0.05)

    for l in lead_time:
        for e in embed_dims:
            embedded_data, target = embed_data(
                lead_time=l,
                embed_dim=e,
                from_date=date(month=2, year=2005, day=1),
                to_date=date(month=1, year=2006, day=31),
                data=data,
                t_idx=2,
                dt_idx=1
            )

            train_x = embedded_data.astype(np.float32)
            train_y = target.astype(np.float32)

            trainX = Variable(torch.from_numpy(train_x)).cuda()
            trainY = Variable(torch.from_numpy(train_y)).cuda()

            embedded_data_test, target_test = embed_data(
                lead_time=l,
                embed_dim=e,
                from_date=date(month=2, year=2006, day=1),
                to_date=date(month=1, year=2007, day=31),
                data=data,
                t_idx=2,
                dt_idx=1
            )

            test_x = embedded_data_test.astype(np.float32)
            test_y = target_test.astype(np.float32)

            testX = Variable(torch.from_numpy(test_x)).cuda()
            testY = Variable(torch.from_numpy(test_y)).cuda()
            crps = []
            for q in tqdm(quantiles):
                regressor = QuantileRegression(
                    input_size=25,
                    hidden_size=40,
                    output_size=1,
                    num_layers=1,
                    seq_length=e,
                    alpha=q
                )

                losses = regressor.train_model(
                    trainX=trainX,
                    trainY=trainY,
                    lr=0.005,
                    epochs=200,
                    sample_size=50,
                    verbose=False
                )

                regressor.eval()
                test_predict = regressor(testX)
                crps.append(regressor.quantile_loss(test_predict, testY).item())
            print("Error for Lead Time : ", l, "Embed Dim: ", e, " RMSE: ", sum(crps) / len(crps))
    regressor.eval()
    train_predict = regressor(trainX)
    data_predict = train_predict.data.cpu().detach().numpy()
    dataY_plot = trainY.data.cpu().detach().numpy()
    print(data_predict.shape, dataY_plot.shape)

    figure(num=None, figsize=(20, 8), dpi=80, facecolor='w', edgecolor='k')
    layers = 1
    from_here = int(data_predict.shape[0] / layers)
    plt.plot(dataY_plot[:200])
    plt.plot(data_predict[:200])

    train_data_reduced, train_target_reduced = embed_data(
        lead_time=6,
        embed_dim=12,
        from_date=date(month=2, year=2006, day=1),
        to_date=date(month=4, year=2006, day=30),
        data=encoded_data,
        t_idx=2,
        dt_idx=1
    )

    regressor_reduced_dim = QuantileRegression(
        input_size=15,
        output_size=1,
        hidden_size=40,
        num_layers=1,
        seq_length=12,
        alpha=0.9
    )

    train_x = train_data_reduced.astype(np.float32)
    train_y = train_target_reduced.astype(np.float32)

    trainX = Variable(torch.from_numpy(train_x))
    trainY = Variable(torch.from_numpy(train_y))

    losses = regressor_reduced_dim.train_model(
        trainX=trainX,
        trainY=trainY,
        lr=0.005,
        epochs=100,
        sample_size=50
    )

    train_data_original, train_target_original = embed_data(
        lead_time=6,
        embed_dim=12,
        from_date=date(month=2, year=2006, day=1),
        to_date=date(month=4, year=2006, day=30),
        data=data,
        t_idx=2,
        dt_idx=1
    )

    regressor_original_dim = QuantileRegression(
        input_size=25,
        output_size=1,
        hidden_size=40,
        num_layers=1,
        seq_length=12,
        alpha=0.9
    )

    train_x = train_data_original.astype(np.float32)
    train_y = train_target_original.astype(np.float32)

    trainX = Variable(torch.from_numpy(train_x))
    trainY = Variable(torch.from_numpy(train_y))

    losses = regressor_original_dim.train_model(
        trainX=trainX,
        trainY=trainY,
        lr=0.005,
        epochs=100,
        sample_size=50
    )

    embedded_data_test, target_test = embed_data(
        lead_time=6,
        embed_dim=12,
        from_date=date(month=5, year=2006, day=1),
        to_date=date(month=6, year=2006, day=30),
        data=data,
        t_idx=2,
        dt_idx=1
    )
    test_x_original = embedded_data_test.astype(np.float32)
    test_y_original = target_test.astype(np.float32)

    testXOriginal = Variable(torch.from_numpy(test_x_original))

    embedded_encoded_data_test, target_test = embed_data(
        lead_time=6,
        embed_dim=12,
        from_date=date(month=5, year=2006, day=1),
        to_date=date(month=6, year=2006, day=30),
        data=encoded_data,
        t_idx=2,
        dt_idx=1
    )

    test_x_reduced = embedded_encoded_data_test.astype(np.float32)
    test_y_reduced = target_test.astype(np.float32)

    testXReduced = Variable(torch.from_numpy(test_x_reduced))

    regressor_original_dim.eval()
    testYOriginalPredict = regressor_original_dim(testXOriginal)
    test_pred_original = testYOriginalPredict.data.numpy()

    regressor_reduced_dim.eval()
    testYReducedPredict = regressor_reduced_dim(testXReduced)
    test_pred_reduced = testYReducedPredict.data.numpy()


    def quantile_loss(output, target, alpha):
        covered_flag = (output <= target)
        uncovered_flag = (output > target)
        return np.mean((target - output) * (alpha) * covered_flag + (output - target) * (1 - alpha) * uncovered_flag)


    lead_time = [6, 12]
    embed_dims = [12, 24, 36]
    quantiles = np.arange(0.05, 1.0, 0.05)

    for l in lead_time:
        for e in embed_dims:
            embedded_data, target = embed_data(
                lead_time=l,
                embed_dim=e,
                from_date=date(month=2, year=2005, day=1),
                to_date=date(month=1, year=2006, day=31),
                data=encoded_data,
                t_idx=2,
                dt_idx=1
            )

            train_x = embedded_data.astype(np.float32)
            train_y = target.astype(np.float32)

            trainX = Variable(torch.from_numpy(train_x)).cuda()
            trainY = Variable(torch.from_numpy(train_y)).cuda()

            embedded_data_test, target_test = embed_data(
                lead_time=l,
                embed_dim=e,
                from_date=date(month=2, year=2006, day=1),
                to_date=date(month=1, year=2007, day=31),
                data=encoded_data,
                t_idx=2,
                dt_idx=1
            )

            test_x = embedded_data_test.astype(np.float32)
            test_y = target_test.astype(np.float32)

            testX = Variable(torch.from_numpy(test_x)).cuda()
            testY = Variable(torch.from_numpy(test_y)).cuda()
            crps = []
            for q in tqdm(quantiles):
                regressor = QuantileRegression(
                    input_size=15,
                    hidden_size=40,
                    output_size=1,
                    num_layers=1,
                    seq_length=e,
                    alpha=q
                )

                losses = regressor.train_model(
                    trainX=trainX,
                    trainY=trainY,
                    lr=0.005,
                    epochs=200,
                    sample_size=50,
                    verbose=False
                )

                regressor.eval()
                predY = regressor(testX).data.cpu().detach().numpy()
                predY = denormalize(predY, mean_out, stds_out)
                testYDenorm = denormalize(test_y, mean_out, stds_out)
                crps.append(quantile_loss(predY, testYDenorm, q))
                print(crps[-1])
            print("Error for Lead Time : ", l, "Embed Dim: ", e, " CRPS: ", sum(crps) / len(crps))






























