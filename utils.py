import os
import re
import random
from itertools import chain
from functools import lru_cache
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


def localize_diagatt(model, X, batch_size=None, max_length=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    batch_size = len(X) if batch_size is None else batch_size
    p = max([X.shape[0] for X in X]) if max_length is None else max_length
    data = MyDataset(X, np.zeros(len(X)), padding=True, device=device, max_length=p)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0)

    outputs_ = []
    with torch.no_grad():
        for batch in dataloader:
            inputs, _ = batch
            outputs = model.attention(inputs)
            outputs = torch.argmax(outputs, dim=-1)
            outputs_.append(outputs.cpu().numpy())
    
    return np.concatenate(outputs_, axis=0)


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


def summarize(tokens, labels, n_tops):
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


def zero_padding(Xs, max_length=None):
    n = len(Xs)
    d = Xs[0].shape[1]
    p = max([X.shape[0] for X in Xs]) if max_length is None else max_length

    data = np.zeros(shape=(n, p, d), dtype=Xs[0].dtype)
    for i, X in enumerate(Xs):
        data[i, :X.shape[0]] = X

    return data


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


class MyDataset(Dataset):
    def __init__(self, Xs, Ys, padding=False, device=None, max_length=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
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


class TextProcessor:
    def __init__(self, model_path='google-bert/bert-large-uncased', device=None, seed=42):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cuda':
            print('\033[34m' + torch.cuda.get_device_name() + '\033[0m')

        set_seed(seed)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        self.d = self.model.config.hidden_size
        self.model.eval()

    @staticmethod
    @lru_cache(maxsize=None)
    def _has_synset(word):
        return bool(wordnet.synsets(word))

    def _vocabulary(self, texts, remove_stopword=True):
        _stopwords = set(stopwords.words('english'))
        _keepwords = ["no", "not"]

        texts = list(map(lambda x: re.sub(r'[^a-z\s]', ' ', x.lower()), texts))
        texts = list(map(lambda x: x.split(), texts))
        vocab = set(chain.from_iterable(texts))
        if remove_stopword:
            vocab = [word for word in vocab if len(word)>2 and len(wordnet.synsets(word))>0]
            vocab = [word for word in vocab if word not in _stopwords]
            vocab.extend(_keepwords)

        self.tokenizer.add_tokens(vocab)
        self.model.resize_token_embeddings(len(self.tokenizer))
        vocab = {token: idx for idx, token in enumerate(vocab)}

        return vocab

    def _embeddings(self, text, vocab, max_length=512, normalized=False):
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=max_length).to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        with torch.no_grad():
            outputs = self.model(**inputs)
            outputs = outputs.last_hidden_state[0]

            tokens_, embeddings_ = [], []
            for i, token in enumerate(tokens):
                if token in vocab.keys():
                    tokens_.append(token)
                    embeddings = outputs[i].detach().cpu().numpy()
                    if normalized:
                        embeddings = embeddings / (np.linalg.norm(embeddings, ord=2) + 1e-15)
                    embeddings_.append(embeddings)

        return np.array(tokens_), np.array(embeddings_)

    def processor(self, texts, vocab=None, max_length=512, normalized=True, remove_stopword=True):
        if vocab is None:
            vocab = self._vocabulary(texts, remove_stopword)

        tokens_, embeddings_ = [], []
        for text in texts:
            tokens, embeddings = self._embeddings(text.lower(), vocab, max_length=max_length, normalized=normalized)
            tokens_.append(tokens)
            embeddings_.append(embeddings)

        vocab = {idx: token for token, idx in vocab.items()}

        return tokens_, embeddings_, vocab

