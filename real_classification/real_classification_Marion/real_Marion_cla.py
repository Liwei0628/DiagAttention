import os 
import gc
import sys
import time
import copy
import warnings
import numpy as np
import pandas as pd
from operator import itemgetter
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
import torch
warnings.filterwarnings("ignore", message="A parameter name that contains `gamma`")
warnings.filterwarnings("ignore", message="A parameter name that contains `beta`")

curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(curr_dir)
sys.path.append(os.path.join(root_dir, '..'))
from utils import cal_accuracy, cal_crsr, get_roi, get_DiagAttroi, get_StdAttcoef, set_seed, TextProcessor
from classMR import GroupLassoRegression, MDSPRegression, TwoStepRegression
from classMR import StdAttRegression, DiagAttRegression, MultiStdAttRegression, MultiDiagAttRegression, RunRegression


# Basic settings
# ## Settings
set_seed(42)
task = 'cla'
n_tops = 5
n_splits = 5
delta = 1e-4
patience = 10
train_size, val_size, test_size = 0.7, 0.1, 0.2   
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# ## Path for checkpoints
checkpoint_path = os.path.join(curr_dir, 'checkpoints')
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

# ## Path for results
result_path = os.path.join(curr_dir, 'results')
if not os.path.exists(result_path):
    os.makedirs(result_path)


# Data
# ## Loading 
train_data = pd.read_csv(os.path.join(curr_dir, 'data', 'train.csv'))
test_data = pd.read_csv(os.path.join(curr_dir, 'data', 'test.csv'))

# ## For ROI
CR_train = train_data['Keyword'].to_numpy().reshape(-1, 1)
CR_test = test_data['Keyword'].to_numpy().reshape(-1, 1)

# ## For X
texts = pd.concat([train_data['Text'], test_data['Text']], axis=0).tolist()

# ### Embedding
start_time = time.time()
Processor = TextProcessor(model_path="google-bert/bert-large-uncased")
Tokens, X, vocab = Processor.processor(texts, normalize=True, remove_stopword=False)
end_time = time.time()
print(f"For Processor | Processing time: {end_time - start_time:.4f} seconds")

# ### Random permutation
permutation = [np.random.permutation(len(X[i])) for i in range(len(X))]
X = [X[i][permutation[i]] for i in range(len(X))]
Tokens = [Tokens[i][permutation[i]] for i in range(len(X))]

# ### Summary
X_train, X_test = X[:len(CR_train)], X[len(CR_train):]
Tokens = pd.DataFrame(Tokens).fillna(0).values
Tokens_train, Tokens_test = Tokens[:len(CR_train)], Tokens[len(CR_train):]

x_dim = (max([len(X[i]) for i in range(len(X))]), Processor.d)
print(f"For X | Vocabulary size: {len(vocab)} | Maximum token count: {x_dim[0]} | Embedding dimension: {x_dim[1]}")

# ## For Y
Y_train = train_data['Label'].to_numpy().reshape(-1, 1)
Y_test = test_data['Label'].to_numpy().reshape(-1, 1)
print(f"Data | Sample Size | Train: {len(Y_train)} | Test: {len(Y_test)}")
print(f"Y | Positive | Train: {Y_train.mean():.3f} | Test: {Y_train.mean():.3f}")

start_time = time.time()


# Group lasso regression (GroupLasso)
# ## Settings for GroupLasso
set_seed(42)
penalty = 'group-lasso'
alpha_alternative = [0.01, 0.1, 1]

# ## Model selection for GroupLasso
CE_GroupLasso, alpha_GroupLasso, GroupLassoClassifier = float("inf"), None, None
for alpha in alpha_alternative:
    CE = 0
    training_params = {'alpha': alpha, 'max_itr': 100, 'tol': 1e-2}

    for train_idx, valid_idx in kf.split(X_train):
        x_train, x_valid = list(itemgetter(*train_idx)(X_train)), list(itemgetter(*valid_idx)(X_train))
        y_train, y_valid = Y_train[train_idx], Y_train[valid_idx]

        Classifier = GroupLassoRegression(x_dim=x_dim, task=task)
        Classifier.fit(x_train, y_train, **training_params)

        y_valid_hat = Classifier.predict(x_valid)
        CE += log_loss(y_valid.ravel(), y_valid_hat.ravel(), normalize=False)
    
    CE = CE / n_splits
    if CE < CE_GroupLasso:
        CE_GroupLasso = CE
        alpha_GroupLasso = alpha
        GroupLassoClassifier = copy.deepcopy(Classifier)

# ## Training for GroupLasso
training_params = {'alpha': alpha_GroupLasso, 'max_itr': 100, 'tol': 1e-2}
GroupLassoClassifier.fit(X_train, Y_train, **training_params)

# ## Evaluation for GroupLasso
# ### For prediction performance 
Y_test_GroupLassoClassifier = GroupLassoClassifier.predict(X_test)
Accuracy_GroupLasso = cal_accuracy(Y_test, Y_test_GroupLassoClassifier)
# ### For ROI identification accuracy
CR_test_GroupLassoClassifier = get_roi(GroupLassoClassifier.coef_[None, :, :], n_tops=n_tops)
CR_test_GroupLassoClassifier = Tokens_test[np.arange(len(Tokens_test))[:, None], CR_test_GroupLassoClassifier]
CRSR_GroupLasso = cal_crsr(CR_test, CR_test_GroupLassoClassifier)

# ## Free memory for GroupLasso
del Classifier, GroupLassoClassifier
torch.cuda.empty_cache()
gc.collect()


# Multidirectional separation penalty (MDSP)
# ## Settings for MDSP
set_seed(42)
penalty = "multidirectional"
alpha_alternative = [0.01, 0.1, 1]

# ## Model selection for MDSP
CE_MDSP, alpha_MDSP, MDSPClassifier= float("inf"), None, None
for alpha in alpha_alternative:
    CE = 0
    training_params = {'alpha': alpha, 'batch_size': 64, 'epochs': 50, 'lr': 1e-4, 'tol': 1e-3}

    for train_idx, valid_idx in kf.split(X_train):
        x_train, x_valid = list(itemgetter(*train_idx)(X_train)), list(itemgetter(*valid_idx)(X_train))
        y_train, y_valid = Y_train[train_idx], Y_train[valid_idx]

        Classifier = MDSPRegression(x_dim=x_dim, task=task)
        Classifier.train(train=(x_train, y_train), **training_params)

        y_valid_hat, _ = Classifier.predict(test=(x_valid, y_valid), **training_params)
        CE += log_loss(y_valid.ravel(), y_valid_hat.ravel(), normalize=False)
    
    CE = CE / n_splits
    if CE < CE_MDSP:
        CE_MDSP = CE
        alpha_MDSP = alpha
        MDSPClassifier = copy.deepcopy(Classifier)

# ## Training for MDSP
training_params = {'alpha': alpha_MDSP, 'batch_size': 64, 'epochs': 50, 'lr': 1e-4, 'tol': 1e-3} 
MDSPClassifier.train(train=(X_train, Y_train), **training_params)

# ## Evaluation for MDSP
# ### For prediction performance 
Y_test_MDSPClassifier, Cs_test_MDSPClassifier = MDSPClassifier.predict(test=(X_test, Y_test), **training_params)
Accuracy_MDSP = cal_accuracy(Y_test, Y_test_MDSPClassifier)
# ### For ROI identification accuracy
CR_test_MDSPClassifier = get_roi(Cs_test_MDSPClassifier, n_tops=n_tops, batch_size=64)
CR_test_MDSPClassifier = Tokens_test[np.arange(len(Tokens_test))[:, None], CR_test_MDSPClassifier]
CRSR_MDSP = cal_crsr(CR_test, CR_test_MDSPClassifier)

# ## Free memory for MDSP
del Classifier, MDSPClassifier, Cs_test_MDSPClassifier
torch.cuda.empty_cache()
gc.collect()


# A two-step method with GMM and Ridge (Two-Step)
# ## Settings for Two-Step
set_seed(42)
penalty = "l2"
n_components = 2
alpha_alternative = [0.01, 0.1, 1]

# ### Training for GMM
training_params = {'covariance_type': 'full', 'max_itr': 100, 'tol': 1e-3}
Classifier = TwoStepRegression(x_dim=x_dim, task=task, n_tops=n_tops, n_components=n_components)
Classifier.fit_gmm(X_train, **training_params)

# ## Model selection for Ridge
CE_TwoStep, alpha_TwoStep, TwoStepClassifier = float("inf"), None, None
for k in range(n_components):
    for alpha in alpha_alternative: 
        CE = 0
        Classifier.k_best = k
        training_params = {'alpha': alpha, 'max_itr': 100, 'tol': 1e-3}

        for train_idx, valid_idx in kf.split(X_train):
            x_train, x_valid = list(itemgetter(*train_idx)(X_train)), list(itemgetter(*valid_idx)(X_train))
            y_train, y_valid = Y_train[train_idx], Y_train[valid_idx]

            classifier = copy.deepcopy(Classifier)
            classifier.fit_reg(x_train, y_train, **training_params)

            _, y_valid_hat = classifier.predict(x_valid)
            CE += log_loss(y_valid.ravel(), y_valid_hat.ravel(), normalize=False)
        
        CE = CE / n_splits
        if CE < CE_TwoStep:
            CE_TwoStep = CE
            alpha_TwoStep = alpha
            TwoStepClassifier = copy.deepcopy(classifier)

# ## Training for Two-Step
training_params = {'alpha': alpha_TwoStep, 'max_itr': 100, 'tol': 1e-3}
TwoStepClassifier.fit_reg(X_train, Y_train, **training_params)

# ## Evaluation for TwoStep
# ### For prediction performance 
CR_test_TwoStepClassifier, Y_test_TwoStepClassifier = TwoStepClassifier.predict(X_test)
Accuracy_TwoStep = cal_accuracy(Y_test, Y_test_TwoStepClassifier)
# ### For ROI identification accuracy
CR_test_TwoStepClassifier = [Tokens_test[i][CR_test_TwoStepClassifier[i]] for i in range(len(Tokens_test))]
CRSR_TwoStep  = cal_crsr(CR_test, CR_test_TwoStepClassifier)

# ## Free memory for TwoStep
del classifier, Classifier, TwoStepClassifier
torch.cuda.empty_cache()
gc.collect()


# Single-head diagonalized attention mechanism (SingleDiagAtt)
# ## Settings for SingleDiagAtt
set_seed(42)
lam = 1e-3
penalty = None
path = os.path.join(checkpoint_path, 'SingleDiagAttClassifier.pt')
training_params = {'batch_size': 64, 'epochs': 50, 'lr': 1e-2, 'opt': 'Adam'} 

# ## Training for SingleDiagAtt
SingleDiagAtt = DiagAttRegression(x_dim=x_dim, lam=lam)
SingleDiagAttClassifier = RunRegression(model=SingleDiagAtt, task=task, patience=patience, delta=delta, path=path)
SingleDiagAttClassifier.train(train=(X_train, Y_train), valid=(X_train, Y_train), **training_params)
W = SingleDiagAttClassifier.model.W.detach().cpu().numpy()

# ## Evaluation for SingleDiagAtt
# ### For prediction performance
Y_test_SingleDiagAttClassifier = SingleDiagAttClassifier.predict(X_test, batch_size=64)
Accuracy_SingleDiagAtt = cal_accuracy(Y_test.ravel(), Y_test_SingleDiagAttClassifier.ravel())
# ### For ROI recovery
CR_test_SingleDiagAttClassifier = get_DiagAttroi(X_test, W, batch_size=64).reshape(-1, 1)
CR_test_SingleDiagAttClassifier = Tokens_test[np.arange(len(Tokens_test))[:, None], CR_test_SingleDiagAttClassifier]
CRSR_SingleDiagAtt = cal_crsr(CR_test, CR_test_SingleDiagAttClassifier)
        
# ## Free memory for SingleDiagAtt
del W, SingleDiagAtt, SingleDiagAttClassifier
torch.cuda.empty_cache()
gc.collect()


# Multi-head diagonalized attention mechanism (MultiDiagAtt)
# ## Settings for MultiDiagAtt
set_seed(42)
lam = 1e-3
penalty = None
n_heads_alternative = [2, 3]
path = os.path.join(checkpoint_path, 'MultiDiagAttClassifier.pt')
training_params = {'batch_size': 64, 'epochs': 50, 'lr': 1e-2, 'opt': 'Adam'} 

# ## Model selection for MultiDiagAtt
CE_MultiDiagAtt, n_heads_MultiDiagAtt, MultiDiagAttClassifier = float("inf"), None, None
for n_heads in n_heads_alternative:
    CE = 0
    for train_idx, valid_idx in kf.split(X_train):
        x_train, x_valid = list(itemgetter(*train_idx)(X_train)), list(itemgetter(*valid_idx)(X_train))
        y_train, y_valid = Y_train[train_idx], Y_train[valid_idx]
        
        MultiDiagAtt = MultiDiagAttRegression(x_dim=x_dim, lam=lam, n_heads=n_heads)
        Classifier = RunRegression(model=MultiDiagAtt, task=task, patience=patience, delta=delta, path=path)
        Classifier.train(train=(x_train, y_train), valid=(x_train, y_train), **training_params)

        y_valid_hat = Classifier.predict(x_valid, batch_size=64)
        CE += log_loss(y_valid.ravel(), y_valid_hat.ravel(), normalize=False)
    
    CE = CE / n_splits
    if CE < CE_MultiDiagAtt:
        CE_MultiDiagAtt = CE
        n_heads_MultiDiagAtt = n_heads
        MultiDiagAttClassifier = copy.deepcopy(Classifier)

# ## Training for MultiDiagAtt
MultiDiagAttClassifier.train(train=(X_train, Y_train), valid=(X_train, Y_train), **training_params)
Ws = [MultiDiagAttClassifier.model.attentions[i].W.detach().cpu().numpy() for i in range(n_heads_MultiDiagAtt)]

# ## Evaluation for MultiDiagAtt
# ### For prediction performance 
Y_test_MultiDiagAttClassifier = MultiDiagAttClassifier.predict(X_test, batch_size=64)
Accuracy_MultiDiagAtt = cal_accuracy(Y_test, Y_test_MultiDiagAttClassifier)
# ### For ROI identification accuracy
CR_test_MultiDiagAttClassifier = [get_DiagAttroi(X_test, Ws[i], batch_size=64) for i in range(n_heads_MultiDiagAtt)]
CR_test_MultiDiagAttClassifier = np.array(CR_test_MultiDiagAttClassifier).T
CR_test_MultiDiagAttClassifier = Tokens_test[np.arange(len(Tokens_test))[:, None], CR_test_MultiDiagAttClassifier]
CRSR_MultiDiagAtt = cal_crsr(CR_test, CR_test_MultiDiagAttClassifier)

# ## Free memory for MultiDiagAtt
del Ws, MultiDiagAtt, Classifier, MultiDiagAttClassifier
torch.cuda.empty_cache()
gc.collect()


# Single-head standard attention mechanism (SingleStdAtt)
# ## Settings for SingleStdAtt
set_seed(42)
penalty = None
path = os.path.join(checkpoint_path, 'SingleStdAttClassifier.pt')
training_params = {'batch_size': 128, 'epochs': 50, 'lr': 1e-4, 'opt': 'Adam'}
        
# ## Training for SingleStdAtt
SingleStdAtt = StdAttRegression(x_dim=x_dim)
SingleStdAttClassifier = RunRegression(model=SingleStdAtt, task=task, patience=patience, delta=delta, path=path)
SingleStdAttClassifier.train(train=(X_train, Y_train), valid=(X_train, Y_train), **training_params)
        
# ## Evaluation for SingleStdAtt
# ### For prediction performance
Y_test_SingleStdAttClassifier = SingleStdAttClassifier.predict(X_test, batch_size=64)
Accuracy_SingleStdAtt = cal_accuracy(Y_test.ravel(), Y_test_SingleStdAttClassifier.ravel())
# ### For ROI recovery
Cs_test_SingleStdAttClassifier = get_StdAttcoef(X_test, SingleStdAttClassifier.model, batch_size=64)
CR_test_SingleStdAttClassifier = get_roi(Cs_test_SingleStdAttClassifier, n_tops=n_tops//n_heads_MultiDiagAtt, batch_size=64)
CR_test_SingleStdAttClassifier = Tokens_test[np.arange(len(Tokens_test))[:, None], CR_test_SingleStdAttClassifier]
CRSR_SingleStdAtt = cal_crsr(CR_test, CR_test_SingleStdAttClassifier)
        
# ## Free memory for SingleStdAtt
del SingleStdAtt, SingleStdAttClassifier, Cs_test_SingleStdAttClassifier
torch.cuda.empty_cache()
gc.collect()


# Multi-head standard attention mechanism (MultiStdAtt)
# ## Settings for MultiStdAtt
set_seed(42)
penalty = None
path = os.path.join(checkpoint_path, 'MultiStdAttClassifier.pt')
training_params = {'batch_size': 128, 'epochs': 50, 'lr': 1e-4, 'opt': 'Adam'}

# ## Training for MultiStdAtt
MultiStdAtt = MultiStdAttRegression(x_dim=x_dim, n_heads=3)
MultiStdAttClassifier = RunRegression(model=MultiStdAtt, task=task, patience=patience, delta=delta, path=path)
MultiStdAttClassifier.train(train=(X_train, Y_train), valid=(X_train, Y_train), **training_params)

# ## Evaluation for MultiStdAtt
# ### For prediction performance
Y_test_MultiStdAttClassifier = MultiStdAttClassifier.predict(X_test, batch_size=64)
Accuracy_MultiStdAtt = cal_accuracy(Y_test.ravel(), Y_test_MultiStdAttClassifier.ravel())
# ### For ROI recovery
Cs_test_MultiStdAttClassifier = [get_StdAttcoef(X_test, attention, batch_size=64) for attention in MultiStdAttClassifier.model.attentions]
CR_test_MultiStdAttClassifier = [get_roi(Cs_test, n_tops=n_tops//n_heads_MultiDiagAtt, batch_size=64) for Cs_test in Cs_test_MultiStdAttClassifier]
CR_test_MultiStdAttClassifier = np.concatenate(CR_test_MultiStdAttClassifier, axis=1)
CR_test_MultiStdAttClassifier = Tokens_test[np.arange(len(Tokens_test))[:, None], CR_test_MultiStdAttClassifier]
CRSR_MultiStdAtt = cal_crsr(CR_test, CR_test_MultiStdAttClassifier)

# ## Free memory for MultiStdAtt
del MultiStdAtt, MultiStdAttClassifier, Cs_test_MultiStdAttClassifier
torch.cuda.empty_cache()
gc.collect()

end_time = time.time()
print(f"Total time: {end_time-start_time:.4f}")


# Summary
Accuracy = [Accuracy_GroupLasso, Accuracy_MDSP, Accuracy_TwoStep, Accuracy_SingleStdAtt, Accuracy_MultiStdAtt, Accuracy_SingleDiagAtt, Accuracy_MultiDiagAtt]
CRSR = [CRSR_GroupLasso, CRSR_MDSP, CRSR_TwoStep, CRSR_SingleStdAtt, CRSR_MultiStdAtt, CRSR_SingleDiagAtt, CRSR_MultiDiagAtt]

table = pd.DataFrame({"Accuracy": Accuracy, "CRSR": CRSR,})
table.index = ['GroupLasso', 'MDSP', 'Two-Step', 'SingleStdAtt', 'MultiStdAtt', 'SingleDiagAtt', 'MultiDiagAtt']
print(table.round(4))
table.to_csv(os.path.join(result_path, f"Performance-All.csv"), index=True)

