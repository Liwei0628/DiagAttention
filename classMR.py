import math
import numpy as np
from scipy.special import expit
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression,LogisticRegression, Ridge
from skglm import GeneralizedLinearEstimator
from skglm.datafits import QuadraticGroup, LogisticGroup
from skglm.penalties import WeightedGroupL2
from skglm.solvers import GroupProxNewton, GroupBCD
from skglm.utils.data import grp_converter
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import set_seed, zero_padding, MyDataset, EarlyStopping


""" Group lasso regression (GroupLasso; Yuan and Lin, 2006) """
class GroupLassoRegression:
    def __init__(self, x_dim, task='reg'):
        self.task = task
        self.coef_ = None
        self.p, self.d = tuple(map(int, x_dim))
    
    def fit(self, X, Y, alpha, max_itr, tol=1e-4, mask=1e-15):
        X = zero_padding(X, max_length=self.p)
        X = X.reshape(len(X), -1) + mask

        grp_indices, grp_ptr = grp_converter(self.d, n_features=self.p*self.d)
        if self.task == 'reg':
            solver = GroupBCD(ws_strategy="fixpoint", fit_intercept=False, max_iter=max_itr, tol=tol, verbose=0)
            model = GeneralizedLinearEstimator(datafit=QuadraticGroup(grp_ptr, grp_indices), solver=solver,
                                               penalty=WeightedGroupL2(alpha, np.ones(self.p), grp_ptr, grp_indices))
        elif self.task == 'cla':
            solver = GroupProxNewton(fit_intercept=False, max_iter=max_itr, tol=tol, verbose=0)
            model = GeneralizedLinearEstimator(datafit=LogisticGroup(grp_ptr, grp_indices), solver=solver,
                                               penalty=WeightedGroupL2(alpha, np.ones(self.p), grp_ptr, grp_indices))
        else:
            raise NotImplementedError(f"Task '{self.task}' not implemented")
        
        model.fit(X, Y.ravel())
        coef_ = model.coef_.reshape(self.p, self.d)

        self.coef_ = coef_

        return 
        
    def predict(self, X):
        if self.coef_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() before predict().")
        
        X = zero_padding(X, max_length=self.p)
        output = np.einsum('npd,pd->n', X, self.coef_)
        output = output if self.task == 'reg' else expit(output)
        output = np.reshape(output, (-1, 1))

        return output


""" Multidirectional separation penalty (MDSP; Tang et al., 2021) """
class MDSPRegression:
    def __init__(self, x_dim, task='reg', device=None):
        if device is None:
            device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        if device.type == 'cuda':
            print('\033[34m' + torch.cuda.get_device_name() + '\033[0m')

        self.device = device
        self.task = task
        self.gamma = None
        self.p, self.d = tuple(map(int, x_dim))
    
    def proximal_threshold(self, v, gamma, weights, threshold):
        # soft-threshold for ||C_{i,j,\cdot}||
        u0 = torch.sign(v) * torch.clamp(torch.abs(v) - threshold, min=0.0)

        # soft-threshold for ||C_{i,j,\cdot}-gamma||
        v_gamma = v - gamma.unsqueeze(0)
        u_gamma = torch.sign(v_gamma) * torch.clamp(torch.abs(v_gamma) - threshold, min=0.0) + gamma.unsqueeze(0)
    
        # Final soft-threshold
        weights = weights.unsqueeze(-1)
        u_soft = torch.where(weights > 0.5, u0, u_gamma)

        return u_soft
    
    def weighted_median(self, u, weights):
        total =  weights.sum(dim=0, keepdim=True)
        half = total / 2

        weights = weights.unsqueeze(2).expand_as(u).contiguous()
        weights = weights.view(-1, self.p*self.d)
        u = u.view(-1, self.p*self.d)

        sorted_u, sort_idx = torch.sort(u, dim=0)
        sorted_weights = torch.gather(weights, 0, sort_idx)
        cum_weights = torch.cumsum(sorted_weights, dim=0)

        half = half.repeat_interleave(self.d, dim=1)
        mask = cum_weights >= half
        median_idx = torch.argmax(mask.int(), dim=0)

        gamma = torch.gather(sorted_u, 0, median_idx.unsqueeze(0)).squeeze(0)
        gamma = gamma.view(self.p, self.d)

        return gamma
    
    @staticmethod
    def cal_delta(current, before):
        diff = current - before
        delta = torch.norm(diff, p=2).item() / math.sqrt(diff.numel())
        
        return delta
    
    def train(self, train, alpha=0, batch_size=32, lr=1e-3, epochs=10, tol=1e-4, print_gap=10):
        train_data = MyDataset(*train, padding=True, device=self.device, max_length=self.p)
        trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=0)
        criterion = nn.MSELoss(reduction='sum') if self.task == 'reg' else nn.BCEWithLogitsLoss(reduction='sum')

        Cs_train = torch.randn(len(train_data), self.p, self.d)
        nn.init.xavier_uniform_(Cs_train)

        if self.gamma is None:
            gamma = torch.randn(self.p, self.d).to(self.device) 
            nn.init.xavier_uniform_(gamma) 
        else:
            gamma = self.gamma.clone().to(self.device)

        for epoch in range(epochs):
            gamma_temp = gamma.detach().clone()
            train_loss, train_acc, delta = 0.0, 0.0, 0.0
            for batch_idx, batch in enumerate(trainloader):
                inputs, labels = batch
                scale = float(len(labels)) if self.task == 'reg' else 1.0
                indices = torch.arange(batch_idx * batch_size, min((batch_idx + 1) * batch_size, len(train_data)))

                coefficients = Cs_train[indices].detach().clone().to(self.device)
                coefficients.requires_grad_(True)
                coefficients_temp = coefficients.detach().clone()

                # Update weights of penalty
                with torch.no_grad():
                    norm_diff = torch.norm(coefficients-gamma.unsqueeze(0), p=2, dim=-1)
                    norm_coef = torch.norm(coefficients, p=1, dim=-1)
                    weights = (norm_diff > norm_coef) + 0
                
                # Update coefficients and gamma by proximial operator
                outputs = torch.einsum('npd,npd->n', coefficients, inputs).unsqueeze(-1)
                loss = criterion(outputs, labels) / scale

                if coefficients.grad is not None:
                    coefficients.grad.zero_()

                loss.backward()
                with torch.no_grad():
                    # Gradient descent
                    coefficients -= lr * coefficients.grad
                    # Proximal operator
                    coefficients = self.proximal_threshold(coefficients, gamma, weights, alpha*lr)
                    # Weighted median
                    gamma = self.weighted_median(coefficients, 1-weights)
                    # Update original coefficients
                    Cs_train[indices] = coefficients.detach().clone().cpu()
                    # Record delta
                    delta += self.cal_delta(coefficients, coefficients_temp)
            
                # Update error/accuracy
                train_loss += loss.item() * scale
                if self.task == 'cla':
                    y_hat = (torch.sigmoid(outputs) > 0.5).float()
                    train_acc += torch.sum(torch.eq(labels, y_hat).float()).item()

            # Report metrics
            train_loss = np.sqrt(train_loss / len(train_data)) if self.task == 'reg' else train_loss / len(train_data)
            train_acc = train_acc / len(train_data)

            if epoch % print_gap == 0:
                if self.task == 'reg':
                    print('Epoch: {} | Train | Loss: {:.4f}'.format(epoch, train_loss), end='\r')
                if self.task == 'cla':
                    print('Epoch: {} | Train | Loss: {:.4f} | Accuracy: {:.4f}'.format(epoch, train_loss, train_acc), end='\r')
            
            if epoch == epochs - 1:
                if self.task == 'reg':
                    print('Epoch: {} | Train | Loss: {:.4f}'.format(epoch, train_loss))
                if self.task == 'cla':
                    print('Epoch: {} | Train | Loss: {:.4f} | Accuracy: {:.4f}'.format(epoch, train_loss, train_acc))

            # Early stopping
            delta = delta / len(trainloader)
            delta += self.cal_delta(gamma_temp, gamma)
            if delta < tol:
                print("Early stopping")
                break
        
        self.gamma = gamma
        print('Finished Training')

        return Cs_train.detach().clone().cpu().numpy()
    
    def predict(self, test, alpha=0, batch_size=32, lr=1e-3, epochs=10, tol=1e-4, print_gap=10):
        if self.gamma is None:
            raise ValueError("Model has not been fitted yet. Call fit() before predict().")
        
        test_data = MyDataset(*test, padding=True, device=self.device, max_length=self.p)
        testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)
        criterion = nn.MSELoss(reduction='sum') if self.task == 'reg' else nn.BCEWithLogitsLoss(reduction='sum')

        Cs_test = torch.randn(len(test_data), self.p, self.d)
        nn.init.xavier_uniform_(Cs_test)

        for epoch in range(epochs):
            test_loss, test_acc, delta = 0.0, 0.0, 0.0
            for batch_idx, batch in enumerate(testloader):
                inputs, labels = batch
                scale = float(len(labels)) if self.task == 'reg' else 1.0
                indices = torch.arange(batch_idx * batch_size, min((batch_idx + 1) * batch_size, len(test_data)))

                coefficients = Cs_test[indices].detach().clone().to(self.device)
                coefficients.requires_grad_(True)
                coefficients_temp = coefficients.detach().clone()

                # Update weights of penalty
                with torch.no_grad():
                    norm_diff = torch.norm(coefficients-self.gamma.unsqueeze(0), p=1, dim=-1)
                    norm_coef = torch.norm(coefficients, p=1, dim=-1)
                    weights = (norm_diff > norm_coef) + 0

                # Update coefficients by proximal operator
                outputs = torch.einsum('npd,npd->n', coefficients, inputs).unsqueeze(-1)
                loss = criterion(outputs, labels) / scale

                if coefficients.grad is not None:
                    coefficients.grad.zero_()

                loss.backward()
                with torch.no_grad():
                    # Gradient descent
                    coefficients -= lr * coefficients.grad
                    # Proximal operator
                    coefficients = self.proximal_threshold(coefficients, self.gamma, weights, alpha*lr)
                    # Update original coefficients
                    Cs_test[indices] = coefficients.detach().clone().cpu()
                    # Record change value
                    delta += self.cal_delta(coefficients, coefficients_temp)
                
                # Update error and accuracy
                test_loss += loss.item() * scale
                if self.task == 'cla':
                    y_hat = (torch.sigmoid(outputs) > 0.5).float()
                    test_acc += torch.sum(torch.eq(labels, y_hat).float()).item()
            
            # Report metrics
            test_loss = np.sqrt(test_loss / len(test_data)) if self.task == 'reg' else test_loss / len(test_data)
            test_acc = test_acc / len(test_data)

            if epoch % print_gap == 0:
                if self.task == 'reg':
                    print('Epoch: {} | Test | Loss: {:.4f}'.format(epoch, test_loss), end='\r')
                if self.task == 'cla':
                    print('Epoch: {} | Test | Loss: {:.4f} | Accuracy: {:.4f}'.format(epoch, test_loss, test_acc), end='\r')
                
            if epoch == epochs - 1:
                if self.task == 'reg':
                    print('Epoch: {} | Test | Loss: {:.4f}'.format(epoch, test_loss))
                if self.task == 'cla':
                    print('Epoch: {} | Test | Loss: {:.4f} | Accuracy: {:.4f}'.format(epoch, test_loss, test_acc))
            
            # Early stopping
            delta = delta / len(testloader)
            if delta < tol:
                print("Early stopping")
                break
        
        outputs_ = []
        for batch_idx, batch in enumerate(testloader):
            inputs, _ = batch
            indices = torch.arange(batch_idx * batch_size, min((batch_idx + 1) * batch_size, len(test_data)))
            coefficients = Cs_test[indices].to(self.device)

            outputs = torch.einsum('npd,npd->n', coefficients, inputs).unsqueeze(-1)
            outputs = torch.sigmoid(outputs) if self.task == 'cla' else outputs
            outputs = outputs.detach().clone().cpu().numpy()
            outputs_.append(outputs)
        
        outputs_ = np.concatenate(outputs_, axis=0)
        Cs_test = Cs_test.detach().clone().cpu().numpy()
        print('Finished Predictions')

        return outputs_, Cs_test


""" A two-step method with Gaussian mixture models and ridge regression (Two-Step; An intuitive method) """
class TwoStepRegression:
    def __init__(self, x_dim, task='reg', n_tops=5, n_components=2):
        self.gmm = None
        self.task = task
        self.coef_ = None
        self.k_best = None
        self.n_tops = n_tops
        self.n_components = n_components
        self.p, self.d = tuple(map(int, x_dim))
    
    def fit_gmm(self, X, covariance_type='full', max_itr=200, tol=1e-3, seed=42):
        X_tilde = np.concatenate(X, axis=0)

        setting = {'n_components': self.n_components, 'covariance_type': covariance_type, 'init_params': "random"}
        gmm = GaussianMixture(**setting, max_iter=max_itr, tol=tol, random_state=seed)
        gmm.fit(X_tilde)
        self.gmm = gmm

        return
    
    def fit_reg(self, X, Y, alpha, max_itr=200, tol=1e-4, seed=42):
        if self.k_best is None:
            raise ValueError("The optimal Gaussian mixture component has not been determined yet.")

        probs = [self.gmm.predict_proba(x)[:, self.k_best] for x in X]
        CR_best = [np.argpartition(prob, -min(self.n_tops, len(prob)))[-min(self.n_tops, len(prob)):] for prob in probs]

        X_tilde = [X[i][CR_best[i]] for i in range(len(X))]
        X_tilde = zero_padding(X_tilde, max_length=self.n_tops)
        X_tilde = np.reshape(X_tilde, (len(X), -1))

        if self.task == 'reg':
            model = LinearRegression(fit_intercept=False) if alpha == 0 else \
                    Ridge(alpha=alpha, fit_intercept=False,  max_iter=max_itr, tol=tol, random_state=seed)
        elif self.task == 'cla':
            model = LogisticRegression(penalty=None, solver='lbfgs', fit_intercept=False,
                                         max_iter=max_itr, tol=tol,  random_state=seed) if alpha ==0 else \
                    LogisticRegression(C=1/alpha, penalty='l2', solver='liblinear',
                                        max_iter=max_itr, tol=tol, fit_intercept=False, random_state=seed)
        else:
            raise NotImplementedError(f"Task '{self.task}' not implemented")

        model.fit(X_tilde, Y.ravel())
        coef_ = model.coef_.reshape(self.n_tops, self.d)

        self.coef_ = coef_

        return
    
    def predict(self, X):
        if self.gmm is None or self.k_best is None or self.coef_ is None:
            raise ValueError("Model has not been fitted yet. Call fit_gmm() and fig_reg() before predict().")

        probs = [self.gmm.predict_proba(x)[:, self.k_best] for x in X]
        CR = [np.argpartition(prob, -min(self.n_tops, len(prob)))[-min(self.n_tops, len(prob)):] for prob in probs]

        X_tilde = [X[i][CR[i]] for i in range(len(X))]
        X_tilde = zero_padding(X_tilde, max_length=self.n_tops)

        output = np.einsum('nsd,sd->n', X_tilde, self.coef_)
        output = output if self.task == 'reg' else expit(output)
        output = np.reshape(output, (-1, 1))

        return CR, output


""" A standard attention mechanism version of individualized regression (StdAtt; Yang et al., 2024) """
class StdAttRegression(nn.Module):
    def __init__(self, x_dim, seed=42):
        super(StdAttRegression, self).__init__()
        set_seed(seed)
        self.p, self.d = tuple(map(int, x_dim))

        self.C = nn.Parameter(torch.randn(self.p, self.d))
        self.W = nn.Parameter(torch.randn(self.d, self.d))
        nn.init.xavier_uniform_(self.C)
        nn.init.xavier_uniform_(self.W)
    
    def cal_coefficients(self, X):
        # Standard attention mechanism
        attention = X @ self.W @ X.transpose(-2, -1) / math.sqrt(self.d)
        attention = F.softmax(attention, dim=-1)

        # Sample-specific coefficients
        coefficients = attention.transpose(-2, -1) @ self.C

        return coefficients
    
    def forward(self, X):
        # Standard attention mechanism
        attention = X @ self.W @ X.transpose(-2, -1) / math.sqrt(self.d)
        attention = F.softmax(attention, dim=-1)

        # Attention-boosted individualized regression
        output = attention @ X
        output = torch.einsum('npd,pd->n', output, self.C)
        output = output.view(-1, 1)

        return output


class MultiStdAttRegression(nn.Module):
    def __init__(self, x_dim, n_heads, seed=42):
        super(MultiStdAttRegression, self).__init__()
        set_seed(seed)
        self.n_heads = n_heads
        self.p, self.d = tuple(map(int, x_dim))
        self.attentions = nn.ModuleList([StdAttRegression(x_dim, seed=seed) for _ in range(n_heads)])
    
    def forward(self, X):
        output = [attention(X) for attention in self.attentions]
        output = torch.stack(output, dim=0)
        output = torch.sum(output, dim=0)

        return output


""" Individualized regression with diagonalized attention mechanism (DiagAtt; Our proposed method) """
class DiagAttRegression(nn.Module):
    def __init__(self, x_dim, lam=1, seed=42):
        super(DiagAttRegression, self).__init__()
        set_seed(seed)
        self.lam = lam
        self.p, self.d = tuple(map(int, x_dim))

        self.v = nn.Parameter(torch.randn(self.d, 1))
        self.W = nn.Parameter(torch.randn(self.d, self.d))
        nn.init.xavier_uniform_(self.v)
        nn.init.xavier_uniform_(self.W)
    
    def forward(self, X):
        # Diagonalized attention mechanism
        attention = X @ self.W @ X.transpose(-2, -1)
        attention = torch.diagonal(attention, dim1=-2, dim2=-1)
        attention = attention.masked_fill(attention == 0, -torch.inf)   ## Mask zero-padded positions (ignored if no zero-padding)
        attention = F.softmax(attention * self.lam, dim=-1)

        # Individualized regression
        output = torch.einsum('np,npd->nd', attention, X)
        output = output @ self.v

        return output


class MultiDiagAttRegression(nn.Module):
    def __init__(self, x_dim, n_heads, lam=1, seed=42):
        super(MultiDiagAttRegression, self).__init__()
        set_seed(seed)
        self.lam = lam
        self.n_heads = n_heads
        self.p, self.d = tuple(map(int, x_dim))
        self.attentions = nn.ModuleList([DiagAttRegression(x_dim, lam=lam, seed=seed) for _ in range(n_heads)])
    
    def forward(self, X):
        output = [attention(X) for attention in self.attentions]
        output = torch.stack(output, dim=0)
        output = torch.sum(output, dim=0)

        return output
        

""" A combination of trainer and predictor for all nonlinear models """
class RunRegression:
    def __init__(self, model, task='reg', patience=10, device=None, delta=0, path='checkpoint.pt'):
        if device is None:
            device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cuda':
            print('\033[34m' + torch.cuda.get_device_name() + '\033[0m')
        
        self.task = task
        self.device = device
        self.model = model.to(self.device)
        self.p, self.d = self.model.p, self.model.d
        self.early_stopping = EarlyStopping(patience=patience, delta=delta, path=path, verbose=False)
    
    def train(self, train, valid, batch_size=32, lr=1e-3, epochs=10, print_gap=10, opt='Adam'):
        train_data = MyDataset(*train, padding=True, device=self.device, max_length=self.p)
        trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)

        valid_data = MyDataset(*valid, padding=True, device=self.device, max_length=self.p)
        validloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True, num_workers=0)

        criterion = nn.MSELoss(reduction='sum') if self.task == 'reg' else nn.BCEWithLogitsLoss(reduction='sum')
        optimizer = optim.Adam(self.model.parameters(), lr=lr) if opt == 'Adam' else optim.SGD(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            train_loss, train_acc = 0.0, 0.0
            self.model.train()
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

            valid_loss, valid_acc = 0.0, 0.0
            self.model.eval()
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
    
    def predict(self, X_test, batch_size=None):
        batch_size = len(X_test) if batch_size is None else batch_size
        test_data = MyDataset(X_test, np.zeros(len(X_test)), padding=True, device=self.device, max_length=self.p)
        testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

        self.model.eval()
        outputs_ = []
        for batch in testloader:
            inputs, _ = batch
            outputs = self.model(inputs)
            outputs = torch.sigmoid(outputs) if self.task == 'cla' else outputs
            outputs = outputs.detach().clone().cpu().numpy()
            outputs_.append(outputs)
        
        return np.concatenate(outputs_, axis=0)

    