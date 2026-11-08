import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from utils import set_seed, MyDataset, EarlyStopping


class DiagAttRegression(nn.Module):
    def __init__(self, x_dim, tau, seed=42):
        super(DiagAttRegression, self).__init__()
        set_seed(seed)
        self.tau = tau
        self.p, self.d = tuple(map(int, x_dim))

        self.v = nn.Parameter(torch.empty(self.d, 1))
        self.W = nn.Parameter(torch.empty(self.d, self.d))
        nn.init.xavier_uniform_(self.v)
        nn.init.xavier_uniform_(self.W)

    def attention(self, X):
        scores = torch.sum((X @ self.W) * X, dim=-1)
        scores = scores.masked_fill(scores == 0, -torch.inf)
        scores = F.softmax(scores / self.tau, dim=-1)

        return scores

    def forward(self, X):
        scores = self.attention(X)
        outputs = torch.einsum('np,npd->nd', scores, X)
        outputs = outputs @ self.v

        return outputs


class MultiDiagAttRegression(nn.Module):
    def __init__(self, x_dim, n_heads, tau, seed=42):
        super(MultiDiagAttRegression, self).__init__()
        set_seed(seed)
        self.tau = tau
        self.n_heads = n_heads
        self.p, self.d = tuple(map(int, x_dim))
        self.components = nn.ModuleList([DiagAttRegression(x_dim, tau=self.tau, seed=seed+i) for i in range(n_heads)])

    def forward(self, X):
        outputs = [component(X) for component in self.components]
        outputs = torch.stack(outputs, dim=0)
        outputs = torch.sum(outputs, dim=0)

        return outputs


class RunRegression:
    def __init__(self, model, task='reg', patience=10, device=None, delta=0, path='checkpoint.pt'):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cuda':
            print('\033[34m' + torch.cuda.get_device_name() + '\033[0m')

        self.task = task
        self.device = device
        self.model = model.to(self.device)
        self.p, self.d = self.model.p, self.model.d
        self.early_stopping = EarlyStopping(patience=patience, delta=delta, path=path, verbose=False)

    def train(self, train, valid, batch_size=32, lr=1e-3, epochs=10, print_gap=10, opt='Adam'):
        train_data = MyDataset(*train, padding=True, device=self.device, max_length=self.p)
        valid_data = MyDataset(*valid, padding=True, device=self.device, max_length=self.p)
        trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
        validloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=0)
        criterion = nn.MSELoss(reduction='sum') if self.task == 'reg' else nn.BCEWithLogitsLoss(reduction='sum')
        optimizer = optim.Adam(self.model.parameters(), lr=lr) if opt == 'Adam' else optim.SGD(self.model.parameters(), lr=lr)  

        for epoch in range(epochs):
            self.model.train()
            train_loss, train_acc = 0.0, 0.0
            for batch in trainloader:
                inputs, labels = batch
                outputs = self.model(inputs)

                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                if self.task == 'cla':
                    y_hat = (torch.sigmoid(outputs) > 0.5).float()
                    train_acc += torch.sum(torch.eq(labels, y_hat).float()).item()

            train_loss = np.sqrt(train_loss / len(train_data)) if self.task == 'reg' else train_loss / len(train_data)
            train_acc = train_acc / len(train_data)

            self.model.eval()
            valid_loss, valid_acc = 0.0, 0.0
            with torch.no_grad():
                for batch in validloader:
                    inputs, labels = batch
                    outputs = self.model(inputs)

                    loss = criterion(outputs, labels)
                    valid_loss += loss.item()
                    if self.task == 'cla':
                        y_hat = (torch.sigmoid(outputs) > 0.5).float()
                        valid_acc += torch.sum(torch.eq(labels, y_hat).float()).item()

            valid_loss = np.sqrt(valid_loss / len(valid_data)) if self.task == 'reg' else valid_loss / len(valid_data)
            valid_acc = valid_acc / len(valid_data)

            if epoch % print_gap == 0:
                if self.task == 'reg':
                    print('Epoch: {} | Loss | Train: {:.4f}, Valid: {:.4f}'.format(epoch, train_loss, valid_loss), end='\r')
                if self.task == 'cla':
                    print('Epoch: {} | Loss | Train {:.4f}, Valid: {:.4f}'.format(epoch, train_loss, valid_loss), end=' | ')
                    print('Accuracy | Train {:.4f}, Valid: {:.4f}'.format(train_acc, valid_acc), end='\r')

            if epoch == epochs - 1:
                if self.task == 'reg':
                    print('Epoch: {} | Loss | Train: {:.4f}, Valid: {:.4f}'.format(epoch, train_loss, valid_loss))
                if self.task == 'cla':
                    print('Epoch: {} | Loss | Train {:.4f}, Valid: {:.4f}'.format(epoch, train_loss, valid_loss), end=' | ')
                    print('Accuracy | Train {:.4f}, Valid: {:.4f}'.format(train_acc, valid_acc))

            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop:
                print("Early stopping")
                break

        print('Finished Training')

        return

    def predict(self, X, batch_size=None):
        batch_size = len(X) if batch_size is None else batch_size
        test_data = MyDataset(X, np.zeros(len(X)), padding=True, device=self.device, max_length=self.p)
        testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

        outputs_ = []
        self.model.eval()
        with torch.no_grad():
            for batch in testloader:
                inputs, _ = batch
                outputs = self.model(inputs)
                outputs = torch.sigmoid(outputs) if self.task == 'cla' else outputs
                outputs_.append(outputs.cpu().numpy())

        return np.concatenate(outputs_, axis=0)

