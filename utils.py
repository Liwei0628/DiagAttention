import os
import re
import random
from itertools import chain
from operator import itemgetter
from collections import Counter
import numpy as np
from nltk.corpus import wordnet, stopwords
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer


def cal_accuracy(y_true, y_hat, bound=0.5):
    y_hat = (y_hat > bound) + 0
    y_true, y_hat = y_true.ravel(), y_hat.ravel()

    return np.sum(y_true == y_hat) / len(y_true)


def cal_crsr(y_true, y_hat):
    y_true = np.array([set(row) for row in y_true], dtype=object)
    y_hat = np.array([set(row) for row in y_hat], dtype=object)
    crsr = [len(y_true[i] & y_hat[i]) / len(y_true[i]) for i in range(len(y_true))]  # Recall

    return np.mean(crsr)


def my_split(CRs, Xs, Ys, train_size, val_size, test_size, shuffle=True, random_state=None):
    # Validate input parameters
    if train_size + val_size + test_size != 1:
        raise ValueError("'train_size + val_size + test_size' must be 1.")
    
    # Train vs. (Validation + Test)
    settings = {'train_size': train_size, 'shuffle': shuffle, 'random_state': random_state}
    CRs_train, CRs_val_test, Xs_train, Xs_val_test, Ys_train, Ys_val_test = train_test_split(CRs, Xs, Ys, **settings)

    # Validation vs. Test
    r = test_size / (val_size + test_size)
    settings = {'test_size': r, 'shuffle': shuffle, 'random_state': random_state}
    CRs_val, CRs_test, Xs_val, Xs_test, Ys_val, Ys_test = train_test_split(CRs_val_test, Xs_val_test, Ys_val_test, **settings)

    return CRs_train, CRs_val, CRs_test, Xs_train, Xs_val, Xs_test, Ys_train, Ys_val, Ys_test


def get_keywords(tokens, labels, n_tops):
    # Positive
    pos = np.where(labels == 1)[0]
    tokens_pos = [list(itemgetter(*pos)(tokens[i])) for i in range(len(tokens))]
    tokens_pos = list(chain.from_iterable(chain.from_iterable(tokens_pos)))
    tokens_pos = np.array(tokens_pos).ravel()
    tokens_pos = tokens_pos[tokens_pos != '0']
    counter_pos = {item: count for item, count in Counter(tokens_pos).most_common(n_tops)}

    # Negative
    neg = np.where(labels == 0)[0]
    tokens_neg = [list(itemgetter(*neg)(tokens[i])) for i in range(len(tokens))]
    tokens_neg = list(chain.from_iterable(chain.from_iterable(tokens_neg)))
    tokens_neg = np.array(tokens_neg).ravel()
    tokens_neg = tokens_neg[tokens_neg != '0']
    counter_neg = {item: count for item, count in Counter(tokens_neg).most_common(n_tops)}

    return counter_pos, counter_neg


def get_roi(Cs, n_tops, batch_size=None, device=None):
    if device is None:
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    batch_size = len(Cs) if batch_size is None else batch_size
    data = MyDataset(Cs, np.zeros(len(Cs)), padding=False, device=device)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0)

    rows_ = []
    for batch in dataloader:
        coefficients, _ = batch
        n, p, d = coefficients.shape
        coefficients = coefficients.view(n*p, d)
    
        rows_norm = torch.norm(coefficients, p=2, dim=-1)
        rows_norm = rows_norm.view(n, p)
        rows = torch.topk(rows_norm, k=n_tops, dim=-1, largest=True, sorted=True).indices.detach().cpu().numpy()
        rows_.append(rows)
        
    return np.concatenate(rows_, axis=0)


def get_DiagAttroi(Xs, W, batch_size=None, max_length=None, device=None):
    if device is None:
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    W = torch.from_numpy(W).float().to(device)
    batch_size = len(Xs) if batch_size is None else batch_size
    p = max([X.shape[0] for X in Xs]) if max_length is None else max_length
    data = MyDataset(Xs, np.zeros(len(Xs)), padding=True, device=device, max_length=p)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0)

    rows_ = []
    with torch.no_grad():
        for batch in dataloader:
            inputs, _ = batch
            attentions = inputs @ W @ inputs.transpose(-2, -1)
            attentions = torch.diagonal(attentions, dim1=-2, dim2=-1)
            attentions = attentions.masked_fill(attentions == 0, -torch.inf)   ##
                
            rows = torch.argmax(attentions, dim=-1)
            rows = rows.detach().clone().cpu().numpy()
            rows_.append(rows)
    
    return np.concatenate(rows_, axis=0)


def get_StdAttcoef(Xs, attention, batch_size=None, device=None):
    if device is None:
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    attention = attention.to(device)
    batch_size = len(Xs) if batch_size is None else batch_size
    data = MyDataset(Xs, np.zeros(len(Xs)), padding=True, device=device)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0)

    coefficients_ = []
    for batch in dataloader:
        inputs, _ = batch
        coefficients = attention.cal_coefficients(inputs)
        coefficients = coefficients.detach().cpu().numpy()
        coefficients_.append(coefficients)
    
    return np.concatenate(coefficients_, axis=0)


def set_seed(seed=666):
    # disable hash randomization and ensure experiment reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def zero_padding(Xs, max_length=None):
    n = len(Xs)
    d = Xs[0].shape[1]
    p = max([X.shape[0] for X in Xs]) if max_length is None else max_length

    data = np.zeros(shape=(n, p, d), dtype=Xs[0].dtype)
    for i, X in enumerate(Xs):
        data[i, :X.shape[0]] = X

    return data


class RegDataGenerator:
    def __init__(self, n_samples, x_dim, n_targets, mu0s, sigma0s, rho=0, device=None, **kwargs):
        # Validate input parameters
        if not isinstance(x_dim, tuple) or len(x_dim) != 2:
            raise ValueError(f"'x_dim' must be a tuple of two integers, but got {len(x_dim)}")
        if not isinstance(mu0s, list) or len(mu0s) != n_targets:
            raise ValueError(f"'mu0s' must be a list of {n_targets} constants, but got {len(mu0s)}")
        if not isinstance(sigma0s, list) or len(sigma0s) != n_targets:
            raise ValueError(f"'sigma0s' must be a list of {n_targets} constants, but got {len(sigma0s)}")
        if n_targets > x_dim[0]:
            raise ValueError("'n_targets' must be less than 'x_dim[0]'")

        # Initialization
        self.n_samples = n_samples
        self.p, self.d = tuple(map(int, x_dim))
        self.n_targets = n_targets
        self.n_nontargets = self.p - self.n_targets

        self.rho = rho
        self.mu0s = mu0s
        self.sigma0s = sigma0s

        self.rng = np.random.default_rng(kwargs.get('seed', 42))
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu') if device is None else device

        # Generate target coefficients
        self.coefficient = self.gnrt_coefficient(size=(self.n_targets, self.d,), **kwargs)

        # Parameters of distribution for X
        self.mu, self.cov = self.gnrt_parameters(**kwargs)
    
    def gnrt_coefficient(self, size, **kwargs):
        mu_coef, sigma_coef = kwargs.get("mu_coef", 1.0), kwargs.get("sigma_coef", 1.0)
        coefficient = self.rng.normal(mu_coef, sigma_coef, size=size)

        return coefficient
    
    def gnrt_parameters(self, **kwargs):
        # Mean for target and non-target
        mu_nontarget = self.rng.normal(0, 0.1, size=(self.d, ))
        nontarget_part = np.tile(mu_nontarget, (self.n_nontargets, ))
        target_part = np.concatenate([mu_nontarget * mu0 for mu0 in self.mu0s], axis=0)
        mu = np.hstack([target_part, nontarget_part])

        # Covariance for target and non-target
        mask = kwargs.get('mask', 1e-4)
        scale = (torch.ones((self.d, self.d)) * self.rho).fill_diagonal_(1)
        scale = torch.kron(scale, torch.ones((self.p, self.p)))

        cov_nontarget = self.rng.normal(size=(self.d, self.d))
        cov_nontarget = cov_nontarget @ cov_nontarget.T + mask * np.eye(self.d)
        std = np.sqrt(np.diag(cov_nontarget))
        cov_nontarget = cov_nontarget / np.outer(std, std)

        eigen_vals, eigen_vecs = torch.linalg.eigh(torch.from_numpy(cov_nontarget).to(self.device))
        cov_nontarget_sqrt = eigen_vecs @ torch.diag(torch.sqrt(eigen_vals)) @ eigen_vecs.T
        nontarget_part = torch.tile(cov_nontarget_sqrt, (self.n_nontargets, 1))

        target_part = []
        for sigma0 in self.sigma0s:
            eigen_vals, eigen_vecs = torch.linalg.eigh(torch.from_numpy(cov_nontarget*sigma0).to(self.device))
            cov_target_sqrt = eigen_vecs @ torch.diag(torch.sqrt(eigen_vals)) @ eigen_vecs.T
            target_part.append(cov_target_sqrt)

        target_part = torch.concatenate(target_part, axis=0)
        cov = torch.vstack([target_part, nontarget_part])
        cov = cov @ cov.T + mask * torch.eye(cov.shape[0], device=self.device)
        cov *= scale.to(cov.device)

        return mu, cov
    
    def gnrt_features(self):
        # Indices for feature vectors
        rows = np.array([self.rng.permutation(self.p) for _ in range(self.n_samples)])  
        rows_target = np.argsort(rows, axis=-1)[:, :self.n_targets]

        # Feature vectors
        cov_sqrt = torch.linalg.cholesky(self.cov)
        feature_vecs = self.rng.normal(size=(self.n_samples, self.d * self.p, ))
        feature_vecs = torch.from_numpy(feature_vecs).to(self.device)

        feature_vecs = feature_vecs @ cov_sqrt.T + torch.from_numpy(self.mu).to(self.device)
        feature_vecs = feature_vecs.detach().cpu().numpy()
        
        feature_vecs = feature_vecs.reshape(self.n_samples, self.p, self.d)
        feature_vecs = feature_vecs[np.arange(self.n_samples)[:, None, None], rows[:, :, None], np.arange(self.d)[None, None, :]]

        return rows_target, feature_vecs
    
    def gnrt_data(self):
        # Features
        rows_target, features = self.gnrt_features()
        features_target = features[np.arange(self.n_samples)[:, None, None], rows_target[:, :, None], np.arange(self.d)[None, None, :]]

        # Labels
        labels = np.einsum('nsd,sd->n', features_target, self.coefficient)
        labels = np.reshape(labels, (-1, 1))
        labels = labels + self.rng.normal(size=[self.n_samples, 1])

        # Summary
        features = np.array(features)
        labels = np.array(labels)
        rows_target = np.array(rows_target)

        return rows_target, features, labels


class ClaDataGenerator:
    def __init__(self, n_samples, x_dim, pi, mu0s, sigma0s, rho=0, device=None, **kwargs):
        # Validate input parameters
        if not isinstance(x_dim, tuple) or len(x_dim) != 2:
            raise ValueError(f"'x_dim' must be a tuple of two integers, but got {len(x_dim)}")
        if not isinstance(mu0s, list) or len(mu0s) != 2:
            raise ValueError(f"'mu0s' must be a list of two constants, but got {len(mu0s)}")
        if not isinstance(sigma0s, list) or len(sigma0s) != 2:
            raise ValueError(f"'sigma0s' must be a list of two constants, but got {len(sigma0s)}")
        
        # Initialization
        self.pi = pi
        self.n_samples = n_samples
        self.p, self.d = tuple(map(int, x_dim))

        self.rho = rho
        self.mu0s = mu0s
        self.sigma0s = sigma0s 

        self.rng = np.random.default_rng(kwargs.get('seed', 42))
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu') if device is None else device


        # Parameters of distribution for X
        self.mus, self.covs = self.gnrt_parameters(**kwargs)
    
    def gnrt_parameters(self, **kwargs):
        # Mean for target and non-target
        mu_nontarget = self.rng.normal(0, 0.1, size=(self.d, ))
        nontarget_part = np.tile(mu_nontarget, (self.p-1, ))
        mus = [np.hstack([mu_nontarget*mu0, nontarget_part]) for mu0 in self.mu0s]

        # Covariance for target and non-target
        mask = kwargs.get('mask', 1e-4)
        scale = (torch.ones((self.d, self.d)) * self.rho).fill_diagonal_(1)
        scale = torch.kron(scale, torch.ones((self.p, self.p)))

        cov_nontarget = self.rng.normal(size=(self.d, self.d))
        cov_nontarget = cov_nontarget @ cov_nontarget.T + mask * np.eye(self.d)
        std = np.sqrt(np.diag(cov_nontarget))
        cov_nontarget = cov_nontarget / np.outer(std, std)

        eigen_vals, eigen_vecs = torch.linalg.eigh(torch.from_numpy(cov_nontarget).to(self.device))
        cov_nontarget_sqrt = eigen_vecs @ torch.diag(torch.sqrt(eigen_vals)) @ eigen_vecs.T
        nontarget_part = torch.tile(cov_nontarget_sqrt, (self.p-1, 1))

        covs = []
        for sigma0 in self.sigma0s:
            eigen_vals, eigen_vecs = torch.linalg.eigh(torch.from_numpy(cov_nontarget*sigma0).to(self.device))
            cov_target_sqrt = eigen_vecs @ torch.diag(torch.sqrt(eigen_vals)) @ eigen_vecs.T

            cov = torch.vstack([cov_target_sqrt, nontarget_part])
            cov = cov @ cov.T + mask * torch.eye(cov.shape[0], device=self.device)
            cov *= scale.to(cov.device)
            covs.append(cov)
        
        return mus, covs
    
    def gnrt_data(self):
        # Labels
        labels = self.rng.binomial(1, self.pi, size=[self.n_samples, 1])
        masks = (labels == 1).ravel()

        # Features
        # ## Indices for feature vectors
        rows = np.array([self.rng.permutation(self.p) for _ in range(self.n_samples)])    
        rows_target = np.argsort(rows, axis=-1)[:, :1]

        # ## Feature vectors
        covs_sqrt = [torch.linalg.cholesky(cov) for cov in self.covs]
        features = self.rng.normal(size=(self.n_samples, self.d * self.p, ))
        features = torch.from_numpy(features).to(self.device)

        features[masks] = features[masks] @ covs_sqrt[0].T + torch.from_numpy(self.mus[0]).to(self.device)
        features[~masks] = features[~masks] @ covs_sqrt[1].T + torch.from_numpy(self.mus[1]).to(self.device)
        features = features.detach().cpu().numpy()

        features = features.reshape(self.n_samples, self.p, self.d)
        features = features[np.arange(self.n_samples)[:, None, None], rows[:, :, None], np.arange(self.d)[None, None, :]]

        # Summary
        features = np.array(features)
        labels = np.array(labels)
        rows_target = np.array(rows_target)

        return rows_target, features, labels
    

class TextProcessor:
    def __init__(self, model_path='google-bert/bert-large-uncased', device=None, seed=42):
        if device is None:
            device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        
        set_seed(seed)
        self.device = device

        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model = self.model.to(self.device)

        self.L = None                               # vocabulary size
        self.d = self.model.config.hidden_size      # embedding dimension
    
    def get_vocabulary(self, texts, remove_stopword=True):
        # Build a vocabulary of stopwords
        dictionary_stopwords = set(stopwords.words('english'))

        # Preprocess sentences: lowercase, remove special characters, split
        texts = list(map(lambda x: re.sub(r'[^a-z\s]', ' ', x.lower()), texts))
        texts = list(map(lambda x: x.split(), texts))

        # Extract vocabulary
        vocab = list(set(chain.from_iterable(texts)))
        if remove_stopword:
            vocab = [word for word in vocab if len(word)>2 and len(wordnet.synsets(word))>0]
            vocab = [word for word in vocab if word not in dictionary_stopwords]
            vocab.extend(['no', 'not'])

        vocab = sorted(vocab)

        # Add new tokens to tokenizer and resize model embeddings
        len_before = len(self.tokenizer)
        self.tokenizer.add_tokens(vocab)
        self.model.resize_token_embeddings(len(self.tokenizer))
        len_after = len(self.tokenizer)
        print(f"Add {len_after-len_before} new tokens")

        # Create vocabulary mapping and set vocabulary size
        vocab = {token: idx for idx, token in enumerate(vocab)}
        self.L = len(vocab)

        return vocab
    
    def get_embeddings(self, text, vocab, max_length=512, normalize=False):
        # Tokenize input text
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=max_length)
        inputs = inputs.to(self.device)

        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        tokens = list(map(lambda x: x[1:] if x.startswith('Ġ') else x, tokens))

        # Generate embeddings with LLM
        with torch.no_grad():
            self.model.eval()
            outputs = self.model(**inputs)
            outputs = outputs.last_hidden_state[0]

            # Get embeddings
            tokens_, embeddings_ = [], []
            for i, token in enumerate(tokens):
                if token in vocab.keys():
                    tokens_.append(token)
                    embedding = outputs[i].detach().cpu().numpy()
                    if normalize:
                        embedding = embedding / np.linalg.norm(embedding, ord=2)
                    embeddings_.append(embedding)
        
        return np.array(tokens_), np.array(embeddings_)
    
    def processor(self, texts, vocab=None, max_length=512, normalize=True, remove_stopword=True):
        if vocab is None:
            vocab = self.get_vocabulary(texts, remove_stopword)
        
        tokens_, embeddings_ = [], []
        for text in texts:
            tokens, embeddings = self.get_embeddings(text.lower(), vocab, max_length=max_length, normalize=normalize)
            tokens_.append(tokens)
            embeddings_.append(embeddings)
        
        vocab = {idx: token for token, idx in vocab.items()}

        return tokens_, embeddings_, vocab


class MyDataset(Dataset):
    def __init__(self, Xs, Ys, padding=False, device=None, max_length=None):
        if device is None:
            device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        
        self.device = device
        self.padding = padding
        self.labels = torch.from_numpy(Ys).float().to(self.device)
        self.features = torch.from_numpy(zero_padding(Xs, max_length)).float().to(self.device) if self.padding else Xs
    
    def __getitem__(self, index):
        label = self.labels[index]
        feature = self.features[index] if self.padding else torch.from_numpy(self.features[index]).float().to(self.device)

        return feature, label
    
    def __len__(self):
        return len(self.labels)


class EarlyStopping:
    def __init__(self, patience=5, delta=0, path='checkpoint.pt', verbose=False):
        """
        Args:
            patience (int): Number of epochs to wait after last time validation loss improved
            delta (float): Minimum change in the monitored quantity to qualify as an improvement
            path (str): Path for saving the model checkpoint
            verbose (bool): Whether to print early stopping messages
        """
        self.patience = patience
        self.delta = delta
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        """Save the current best model"""
        torch.save(model.state_dict(), self.path)
        if self.verbose:
            print(f"Validation loss decreased. Saving model to {self.path}")

