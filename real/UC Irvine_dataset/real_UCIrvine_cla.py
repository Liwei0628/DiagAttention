import os 
import gc
import sys
import time
import copy
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
import torch
warnings.filterwarnings("ignore", message="A parameter name that contains `gamma`")
warnings.filterwarnings("ignore", message="A parameter name that contains `beta`")

curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(curr_dir)
sys.path.append(os.path.join(root_dir, '..'))
from utils import cal_accuracy, get_keywords, get_roi, get_DiagAttroi, get_StdAttcoef, my_split, set_seed, TextProcessor
from classMR import GroupLassoRegression, MDSPRegression, TwoStepRegression
from classMR import StdAttRegression, DiagAttRegression, MultiStdAttRegression, MultiDiagAttRegression, RunRegression

# Basic settings
# ## Settings
set_seed(42)
task = 'cla'
n_tops = 5
delta = 1e-4
patience = 10
n_repeats = 100
train_size, val_size, test_size = 0.7, 0.1, 0.2    

# ## Lists for Accuracy of different methods
Accuracy_GroupLasso, Accuracy_MDSP, Accuracy_TwoStep = [], [], []
Accuracy_SingleStdAtt, Accuracy_MultiStdAtt, Accuracy_SingleDiagAtt, Accuracy_MultiDiagAtt = [], [], [], []

# ## Lists for Tokens of different methods
Tokens_GroupLasso, Tokens_MDSP, Tokens_TwoStep = [], [], []
Tokens_SingleStdAtt, Tokens_MultiStdAtt, Tokens_SingleDiagAtt, Tokens_MultiDiagAtt = [], [], [], []

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
files = ['amazon_cells_labelled.txt', 'imdb_labelled.txt', 'yelp_labelled.txt']
files = list(map(lambda x: os.path.join(curr_dir, 'data', x), files))
df = pd.concat(map(lambda x: pd.read_csv(x, sep='\t', header=None,
                                         names=['text', 'label'], quoting=3), files), ignore_index=True)

# ## For X
texts = df.text.tolist()

start_time = time.time()
Processor = TextProcessor(model_path="google-bert/bert-large-uncased")
Tokens, X, vocab = Processor.processor(texts, normalize=True, remove_stopword=True)
end_time = time.time()
print(f"For Processor | Processing time: {end_time - start_time:.4f} seconds")

x_dim = (max([len(X[i]) for i in range(len(X))]), Processor.d)
Tokens = pd.DataFrame(Tokens).fillna(0).values
print(f"For X | Vocabulary size: {len(vocab)} | Maximum token count: {x_dim[0]} | Embedding dimension: {x_dim[1]}")

# ## For Y
Y = df.label.to_numpy()
Y = Y.reshape(-1, 1)


start_time = time.time()
for i in range(n_repeats):
    # Split data
    settings = {'train_size': train_size, 'val_size': val_size, 'test_size': test_size, 'shuffle': True}  
    _, _, _, x_train, x_valid, X_test, y_train, y_valid, Y_test = my_split(np.zeros(len(X)), X, Y, **settings, random_state=i+1)
    X_train, Y_train = x_train + x_valid, np.concatenate([y_train, y_valid], axis=0)

    print(f"Data | Sample Size | Train: {len(y_train)} | Valid: {len(y_valid)} | Test: {len(Y_test)}")
    print(f'For Y | Positive | Train: {y_train.mean():.3f}, Valid: {y_valid.mean():.3f}, Test: {Y_test.mean():.3f}')


    # Group lasso regression (GroupLasso)
    # ## Settings for GroupLasso
    set_seed(42)
    penalty = 'group-lasso'
    alpha_alternative = [0.01, 0.1, 1]

    # ## Model selection for GroupLasso
    CE_GroupLasso, alpha_GroupLasso, GroupLassoClassifier = float("inf"), None, None
    Classifier = GroupLassoRegression(x_dim=x_dim, task=task)
    for alpha in alpha_alternative:
        # For training
        training_params = {'alpha': alpha, 'max_itr': 100, 'tol': 1e-2}
        Classifier.fit(x_train, y_train, **training_params)

        # For model selection
        y_valid_hat = Classifier.predict(x_valid)
        CE = log_loss(y_valid.ravel(), y_valid_hat.ravel(), normalize=False)

        if CE < CE_GroupLasso:
            CE_GroupLasso = CE
            alpha_GroupLasso = alpha
            GroupLassoClassifier = copy.deepcopy(Classifier)

    # ## Training for GroupLasso
    training_params = {'alpha': alpha_GroupLasso, 'max_itr': 100, 'tol': 1e-2}
    GroupLassoClassifier.fit(X_train, Y_train, **training_params)

    # ## Evaluation for GroupLasso
    Y_test_GroupLassoClassifier = GroupLassoClassifier.predict(X_test)
    Accuracy_GroupLasso.append(cal_accuracy(Y_test, Y_test_GroupLassoClassifier))

    # ## Representative tokens for GroupLasso
    Tokens_GroupLassoClassifier = get_roi(GroupLassoClassifier.coef_[None, :, :], n_tops=n_tops)
    Tokens_GroupLassoClassifier = Tokens[np.arange(len(Tokens))[:, None], Tokens_GroupLassoClassifier]
    Tokens_GroupLasso.append(Tokens_GroupLassoClassifier)

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
    Classifier = MDSPRegression(x_dim=x_dim, task=task)
    for alpha in alpha_alternative:
        # For training
        training_params = {'alpha': alpha, 'batch_size': 64, 'epochs': 50, 'lr': 1e-3, 'tol': 1e-4}  # For hyperparameter selection
        Classifier.train(train=(x_train, y_train), **training_params)

        # For model selection
        y_valid_hat, _ = Classifier.predict(test=(x_valid, y_valid), **training_params)
        CE = log_loss(y_valid.ravel(), y_valid_hat.ravel(), normalize=False)

        if CE < CE_MDSP:
            CE_MDSP = CE
            alpha_MDSP = alpha
            MDSPClassifier = copy.deepcopy(Classifier)

    # ## Training for MDSP
    training_params = {'alpha': alpha_MDSP, 'batch_size': 64, 'epochs': 50, 'lr': 1e-3, 'tol': 1e-4} 
    Cs_train_MDSPClassifier = MDSPClassifier.train(train=(X_train, Y_train), **training_params)

    # ## Evaluation for MDSP
    Y_test_MDSPClassifier, Cs_test_MDSPClassifier = MDSPClassifier.predict(test=(X_test, Y_test), **training_params)
    Accuracy_MDSP.append(cal_accuracy(Y_test, Y_test_MDSPClassifier))

    # ## Representative tokens for MDSP
    Cs_MDSPClassifier = np.concatenate([Cs_train_MDSPClassifier, Cs_test_MDSPClassifier], axis=0)
    Tokens_MDSPClassifier = get_roi(Cs_MDSPClassifier, n_tops=n_tops, batch_size=256)
    Tokens_MDSPClassifier = Tokens[np.arange(len(Tokens))[:, None], Tokens_MDSPClassifier]
    Tokens_MDSP.append(Tokens_MDSPClassifier)

    # ## Free memory for MDSP
    del Classifier, MDSPClassifier, Cs_train_MDSPClassifier, Cs_test_MDSPClassifier, Cs_MDSPClassifier
    torch.cuda.empty_cache()
    gc.collect()


    # A two-step method with GMM and Ridge (Two-Step)
    # ## Settings for Two-Step
    set_seed(42)
    penalty = "l2"
    n_components = 2
    alpha_alternative = [0.01, 0.1, 1]

    # ## Training for GMM
    training_params = {'covariance_type': 'full', 'max_itr': 100, 'tol': 1e-3}
    Classifier = TwoStepRegression(x_dim=x_dim, task=task, n_tops=n_tops, n_components=n_components)
    Classifier.fit_gmm(X_train, **training_params)

    # ## Model selection for Ridge
    CE_TwoStep, alpha_TwoStep, TwoStepClassifier = float("inf"), None, None
    for k in range(n_components):
        Classifier.k_best = k
        for alpha in alpha_alternative:
            # For training
            training_params = {'alpha': alpha, 'max_itr': 100, 'tol': 1e-3}
            Classifier.fit_reg(x_train, y_train, **training_params)

            # For model selection
            _, y_valid_hat = Classifier.predict(x_valid)
            CE = log_loss(y_valid.ravel(), y_valid_hat.ravel(), normalize=False)

            if CE < CE_TwoStep:
                CE_TwoStep = CE
                alpha_TwoStep = alpha
                TwoStepClassifier = copy.deepcopy(Classifier)
    
    # ## Training for Ridge
    training_params = {'alpha': alpha_TwoStep, 'max_itr': 100, 'tol': 1e-3}
    TwoStepClassifier.fit_reg(X_train, Y_train, **training_params)

    # ## Evaluation for TwoStep
    CR_test_TwoStepClassifier, Y_test_TwoStepClassifier = TwoStepClassifier.predict(X_test)
    Accuracy_TwoStep.append(cal_accuracy(Y_test, Y_test_TwoStepClassifier))

    # ## Representative tokens for TwoStep
    Tokens_TwoStepClassifier, _ = TwoStepClassifier.predict(X)
    Tokens_TwoStepClassifier = [Tokens[i][Tokens_TwoStepClassifier[i]] for i in range(len(Tokens))]
    Tokens_TwoStep.append(Tokens_TwoStepClassifier)

    # ## Free memory for TwoStep
    del Classifier, TwoStepClassifier
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
    Y_test_SingleDiagAttClassifier = SingleDiagAttClassifier.predict(X_test, batch_size=64)
    Accuracy_SingleDiagAtt.append(cal_accuracy(Y_test.ravel(), Y_test_SingleDiagAttClassifier.ravel()))

    # ## Representative tokens for SingleDiagAtt
    Tokens_SingleDiagAttClassifier = get_DiagAttroi(X, W, batch_size=64).reshape(-1, 1)
    Tokens_SingleDiagAttClassifier = Tokens[np.arange(len(Tokens))[:, None], Tokens_SingleDiagAttClassifier]
    Tokens_SingleDiagAtt.append(Tokens_SingleDiagAttClassifier)
        
    # ## Free memory for SingleDiagAtt
    del W, SingleDiagAttClassifier
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
    MultiDiagAtt = MultiDiagAttRegression(x_dim=x_dim, lam=lam, n_heads=0)
    MultiDiagAtt.attentions.append(SingleDiagAtt)
    Classifier = RunRegression(model=MultiDiagAtt, task=task, patience=patience, delta=delta, path=path)
    for n_heads in n_heads_alternative:
        # For training
        Classifier.model.attentions.append(DiagAttRegression(x_dim, lam=lam).cuda("cuda:1"))
        Classifier.train(train=(x_train, y_train), valid=(x_train, y_train), **training_params)
    
        # For model selection
        y_valid_hat = Classifier.predict(x_valid, batch_size=64)
        CE = log_loss(y_valid.ravel(), y_valid_hat.ravel(), normalize=False)
    
        if CE < CE_MultiDiagAtt:
            CE_MultiDiagAtt = CE
            n_heads_MultiDiagAtt = n_heads
            MultiDiagAttClassifier = copy.deepcopy(Classifier)
            
    # ## Training for MultiDiagAtt
    MultiDiagAttClassifier.train(train=(X_train, Y_train), valid=(X_train, Y_train), **training_params)
    Ws = [MultiDiagAttClassifier.model.attentions[i].W.detach().cpu().numpy() for i in range(n_heads_MultiDiagAtt)]
    
    # ## Evaluation for MultiDiagAtt
    Y_test_MultiDiagAttClassifier = MultiDiagAttClassifier.predict(X_test, batch_size=64)
    Accuracy_MultiDiagAtt.append(cal_accuracy(Y_test.ravel(), Y_test_MultiDiagAttClassifier.ravel()))

    # ## Representative tokens for MultiDiagAtt
    Tokens_MultiDiagAttClassifier = [get_DiagAttroi(X, Ws[i], batch_size=64) for i in range(n_heads_MultiDiagAtt)]
    Tokens_MultiDiagAttClassifier = np.array(Tokens_MultiDiagAttClassifier).T
    Tokens_MultiDiagAttClassifier = Tokens[np.arange(len(Tokens))[:, None], Tokens_MultiDiagAttClassifier]
    Tokens_MultiDiagAtt.append(Tokens_MultiDiagAttClassifier)
            
    # ## Free memory for MultiDiagAtt
    del Ws, SingleDiagAtt, MultiDiagAtt, Classifier, MultiDiagAttClassifier
    torch.cuda.empty_cache()
    gc.collect()


    # Single-head standard attention mechanism (SingleStdAtt)
    # ## Settings for SingleStdAtt
    set_seed(42)
    penalty = None
    path = os.path.join(checkpoint_path, 'SingleStdAttClassifier.pt')
    training_params = {'batch_size': 256, 'epochs': 50, 'lr': 1e-4, 'opt': 'Adam'}
        
    # ## Training for SingleStdAtt
    SingleStdAtt = StdAttRegression(x_dim=x_dim)
    SingleStdAttClassifier = RunRegression(model=SingleStdAtt, task=task, patience=patience, delta=delta, path=path)
    SingleStdAttClassifier.train(train=(X_train, Y_train), valid=(X_train, Y_train), **training_params)
        
    # ## Evaluation for SingleStdAtt
    Y_test_SingleStdAttClassifier = SingleStdAttClassifier.predict(X_test, batch_size=64)
    Accuracy_SingleStdAtt.append(cal_accuracy(Y_test.ravel(), Y_test_SingleStdAttClassifier.ravel()))

    # ## Representative tokens for SingleStdAtt
    Cs_SingleStdAttClassifier = get_StdAttcoef(X, SingleStdAttClassifier.model, batch_size=64)
    Tokens_SingleStdAttClassifier = get_roi(Cs_SingleStdAttClassifier, n_tops=n_tops, batch_size=64)
    Tokens_SingleStdAttClassifier = Tokens[np.arange(len(Tokens))[:, None], Tokens_SingleStdAttClassifier]
    Tokens_SingleStdAtt.append(Tokens_SingleStdAttClassifier)
        
    # ## Free memory for SingleStdAtt
    del SingleStdAtt, SingleStdAttClassifier, Cs_SingleStdAttClassifier
    torch.cuda.empty_cache()
    gc.collect()


    # Multi-head standard attention mechanism (MultiStdAtt)
    # ## Settings for MultiStdAtt
    set_seed(42)
    penalty = None
    path = os.path.join(checkpoint_path, 'MultiStdAttClassifier.pt')
    training_params = {'batch_size': 256, 'epochs': 50, 'lr': 1e-4, 'opt': 'Adam'}

    # ## Training for MultiStdAtt
    MultiStdAtt = MultiStdAttRegression(x_dim=x_dim, n_heads=n_heads_MultiDiagAtt)
    MultiStdAttClassifier = RunRegression(model=MultiStdAtt, task=task, patience=patience, delta=delta, path=path)
    MultiStdAttClassifier.train(train=(X_train, Y_train), valid=(X_train, Y_train), **training_params)

    # ## Evaluation for MultiStdAtt
    Y_test_MultiStdAttClassifier = MultiStdAttClassifier.predict(X_test, batch_size=64)
    Accuracy_MultiStdAtt.append(cal_accuracy(Y_test.ravel(), Y_test_MultiStdAttClassifier.ravel()))

    # ### Representative tokens for MultiStdAtt
    Cs_MultiStdAttClassifier = [get_StdAttcoef(X, attention, batch_size=64) for attention in MultiStdAttClassifier.model.attentions]
    Tokens_MultiStdAttClassifier = [get_roi(Cs, n_tops=n_tops//n_heads_MultiDiagAtt, batch_size=64) for Cs in Cs_MultiStdAttClassifier]
    Tokens_MultiStdAttClassifier = np.concatenate(Tokens_MultiStdAttClassifier, axis=1)
    Tokens_MultiStdAttClassifier = Tokens[np.arange(len(Tokens))[:, None], Tokens_MultiStdAttClassifier]
    Tokens_MultiStdAtt.append(Tokens_MultiStdAttClassifier)

    # ## Free memory for MultiStdAtt
    del MultiStdAtt, MultiStdAttClassifier, Cs_MultiStdAttClassifier
    torch.cuda.empty_cache()
    gc.collect()

end_time = time.time()
print(f"Total time: {end_time-start_time:.4f} | Average time: {(end_time-start_time)/n_repeats:.4f}")

# Summary
# ## Accuracy
table = pd.DataFrame({'GroupLasso': Accuracy_GroupLasso, 'MDSP': Accuracy_MDSP, 'Two-Step': Accuracy_TwoStep,
                      'SingleStdAtt': Accuracy_SingleStdAtt, 'MultiStdAtt': Accuracy_MultiStdAtt, 'SingleDiagAtt': Accuracy_SingleDiagAtt, 'MultiDiagAtt': Accuracy_MultiDiagAtt})
table.to_csv(os.path.join(result_path, f"performance-Accuracy.csv"), index=False)

# ## Mean & S.D.
Accuracy_mean = [np.mean(Accuracy_GroupLasso), np.mean(Accuracy_MDSP), np.mean(Accuracy_TwoStep), 
                 np.mean(Accuracy_SingleStdAtt), np.mean(Accuracy_MultiStdAtt), np.mean(Accuracy_SingleDiagAtt), np.mean(Accuracy_MultiDiagAtt)]
Accuracy_std = [np.std(Accuracy_GroupLasso), np.std(Accuracy_MDSP), np.std(Accuracy_TwoStep), 
                np.std(Accuracy_SingleStdAtt), np.std(Accuracy_MultiStdAtt), np.std(Accuracy_SingleDiagAtt), np.std(Accuracy_MultiDiagAtt)]

table = pd.DataFrame({"Accuracy_mean": Accuracy_mean, "Accuracy_std": Accuracy_std,})
table.index = ['GroupLasso', 'MDSP', 'Two-Step', 'SingleStdAtt', 'MultiStdAtt', 'SingleDiagAtt', 'MultiDiagAtt']
print(table.round(4))
table.to_csv(os.path.join(result_path, f"performance-All.csv"), index=True)

# ## Tokens
# ### GroupLasso
Positive_GroupLasso, Negative_GroupLasso = get_keywords(Tokens_GroupLasso, Y, n_tops=10)
table = pd.DataFrame({'Positive_Tokens': list(Positive_GroupLasso.keys()), 'Positive_Counters': list(Positive_GroupLasso.values()),
                      'Negative_Tokens': list(Negative_GroupLasso.keys()), 'Negative_Counters': list(Negative_GroupLasso.values())})
table.to_csv(os.path.join(result_path, f"Summary-GroupLasso.csv"), index=True)

# ### MDSP
Positive_MDSP, Negative_MDSP = get_keywords(Tokens_MDSP, Y, n_tops=10)
table = pd.DataFrame({'Positive_Tokens': list(Positive_MDSP.keys()), 'Positive_Counters': list(Positive_MDSP.values()),
                      'Negative_Tokens': list(Negative_MDSP.keys()), 'Negative_Counters': list(Negative_MDSP.values())})
table.to_csv(os.path.join(result_path, f"Summary-MDSP.csv"), index=True)

# ### Two-Step
Positive_TwoStep, Negative_TwoStep = get_keywords(Tokens_TwoStep, Y, n_tops=10)
table = pd.DataFrame({'Positive_Tokens': list(Positive_TwoStep.keys()), 'Positive_Counters': list(Positive_TwoStep.values()),
                      'Negative_Tokens': list(Negative_TwoStep.keys()), 'Negative_Counters': list(Negative_TwoStep.values())})
table.to_csv(os.path.join(result_path, f"Summary-TwoStep.csv"), index=True)

# ### SingleStdAtt
Positive_SingleStdAtt, Negative_SingleStdAtt = get_keywords(Tokens_SingleStdAtt, Y, n_tops=10)
table = pd.DataFrame({'Positive_Tokens': list(Positive_SingleStdAtt.keys()), 'Positive_Counters': list(Positive_SingleStdAtt.values()),
                      'Negative_Tokens': list(Negative_SingleStdAtt.keys()), 'Negative_Counters': list(Negative_SingleStdAtt.values())})
table.to_csv(os.path.join(result_path, f"Summary-SingleStdAtt.csv"), index=True)

# ### MultiStdAtt
Positive_MultiStdAtt, Negative_MultiStdAtt = get_keywords(Tokens_MultiStdAtt, Y, n_tops=10)
table = pd.DataFrame({'Positive_Tokens': list(Positive_MultiStdAtt.keys()), 'Positive_Counters': list(Positive_MultiStdAtt.values()),
                      'Negative_Tokens': list(Negative_MultiStdAtt.keys()), 'Negative_Counters': list(Negative_MultiStdAtt.values())})
table.to_csv(os.path.join(result_path, f"Summary-MultiStdAtt.csv"), index=True)

# ### SingleDiagAtt
Positive_SingleDiagAtt, Negative_SingleDiagAtt = get_keywords(Tokens_SingleDiagAtt, Y, n_tops=10)
table = pd.DataFrame({'Positive_Tokens': list(Positive_SingleDiagAtt.keys()), 'Positive_Counters': list(Positive_SingleDiagAtt.values()),
                      'Negative_Tokens': list(Negative_SingleDiagAtt.keys()), 'Negative_Counters': list(Negative_SingleDiagAtt.values())})
table.to_csv(os.path.join(result_path, f"Summary-SingleDiagAtt.csv"), index=True)

# ### MultiDiagAtt
Positive_MultiDiagAtt, Negative_MultiDiagAtt = get_keywords(Tokens_MultiDiagAtt, Y, n_tops=10)
table = pd.DataFrame({'Positive_Tokens': list(Positive_MultiDiagAtt.keys()), 'Positive_Counters': list(Positive_MultiDiagAtt.values()),
                      'Negative_Tokens': list(Negative_MultiDiagAtt.keys()), 'Negative_Counters': list(Negative_MultiDiagAtt.values())})
table.to_csv(os.path.join(result_path, f"Summary-MultiDiagAtt.csv"), index=True)

