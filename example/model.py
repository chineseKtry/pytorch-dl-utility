from common_defs import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

space = {
    'dropout': hp.choice('dropout', (0.1, 0.5, 0.75)),
    'delta': hp.choice('delta', (1e-04, 1e-06, 1e-08)),
    'momentum': hp.choice('momentum', (0.9, 0.99, 0.999)),
}


def get_params():
    params = sample(space)
    return handle_integers(params)


def print_params(params):
    pprint({k: v for k, v in params.items() if not k.startswith('layer_')})
    print


class Network(nn.Module):

    def __init__(self, params):
        super(Network, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(params['input_shape'][0], 128, (1, 24), padding=(0, 11)),
            nn.ReLU(),
            nn.MaxPool2d((1, 100))
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(params['dropout']),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def get_model(params):
    model = Network(params)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters(), eps=params['delta'], rho=params['momentum'])
    return model, criterion, optimizer


def try_params(params, n_iterations, data=None, datamode='memory'):
    print 'iterations:', n_iterations
    print_params(params)

    batch_size = 100
    if datamode == 'memory':
        X_train, Y_train = data['train']
        X_val, Y_val = data['valid']
        input_shape = X_train.shape[1:]
    else:
        train_generator = data['train']['gen_func'](batch_size, data['train']['path'])
        valid_generator = data['valid']['gen_func'](batch_size, data['valid']['path'])
        train_epoch_step = data['train']['n_sample'] / batch_size
        valid_epoch_step = data['valid']['n_sample'] / batch_size
        input_shape = data['train']['gen_func'](
            batch_size, data['train']['path']).next()[0].shape[1:]
    params['input_shape'] = input_shape

    model, criterion, optimizer = get_model(params)

    if datamode == 'memory':
        for t in range(n_iterations):
            indices = np.random.choice(len(X_train), size=batch_size, replace=False)
            x, y = torch.from_numpy(X_train[indices]), torch.from_numpy(
                np.argmax(Y_train[indices], axis=1))

            y_pred = model(x.float())
            loss = criterion(y_pred, y.long())
            loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        y_pred_logit = model(torch.from_numpy(X_val).float())
        max_pred, y_pred = torch.max(y_pred_logit, 1)
        Y_val_value = torch.from_numpy(np.argmax(Y_val, axis=1)).long()
        loss = criterion(y_pred_logit, Y_val_value)
        # accuracy = y_pred.eq(Y_val_value).float().mean()
        # print(loss.item(), accuracy.item())
    else:
        raise NotImplementedError('Generator not supported for now')
    return dict(loss=loss, params=params)
