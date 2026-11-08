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
sys.path.append(os.path.join(curr_dir, '..'))
from utils import cal_accuracy, localize_diagatt, my_split, set_seed, summarize, TextProcessor
from classAttention import DiagAttRegression, MultiDiagAttRegression, RunRegression


# Basic settings
# ## Settings
set_seed(42)
task = 'cla'
delta = 1e-4
patience = 10
n_repeats = 10
train_size, val_size, test_size = 0.7, 0.1, 0.2 

# ## Lists of different methods
Accuracy_SingleDiagAtt, Accuracy_MultiDiagAtt = [], []
Tokens_SingleDiagAtt, Tokens_MultiDiagAtt = [], []

# ## Path for checkpoints
checkpoint_path = os.path.join(curr_dir, 'checkpoints')
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)


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
Tokens, X, vocab = Processor.processor(texts, normalized=True, remove_stopword=True)
end_time = time.time()
print(f"For Processor | Processing time: {end_time - start_time:.4f} seconds")

x_dim = (max([len(X[i]) for i in range(len(X))]), Processor.d)
Tokens = pd.DataFrame(Tokens).fillna(0).values
print(f"For X | Vocabulary size: {len(vocab)} | Maximum row dimension: {x_dim[0]} | Feature dimension: {x_dim[1]}")

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

    
    # Single-head diagonalized attention mechanism (SingleDiagAtt)
    # ## Settings for SingleDiagAtt
    set_seed(42)
    tau = 1e3
    path = os.path.join(checkpoint_path, 'SingleDiagAttClassifier.pt')
    training_params = {'batch_size': 64, 'epochs': 50, 'lr': 1e-2, 'opt': 'Adam'} 

    # ## Training for SingleDiagAtt
    SingleDiagAtt = DiagAttRegression(x_dim=x_dim, tau=tau)
    SingleDiagAttClassifier = RunRegression(model=SingleDiagAtt, task=task, patience=patience, delta=delta, path=path)
    SingleDiagAttClassifier.train(train=(X_train, Y_train), valid=(X_train, Y_train), **training_params)

    # ## Evaluation for SingleDiagAtt
    Y_test_SingleDiagAttClassifier = SingleDiagAttClassifier.predict(X_test, batch_size=64)
    Accuracy_SingleDiagAtt.append(cal_accuracy(Y_test.ravel(), Y_test_SingleDiagAttClassifier.ravel()))

    # ## Representative tokens for SingleDiagAtt
    Tokens_SingleDiagAttClassifier = localize_diagatt(SingleDiagAttClassifier.model, X, batch_size=64).reshape(-1, 1)
    Tokens_SingleDiagAttClassifier = Tokens[np.arange(len(Tokens))[:, None], Tokens_SingleDiagAttClassifier]
    Tokens_SingleDiagAtt.append(Tokens_SingleDiagAttClassifier)

    # ## Free memory for SingleDiagAtt
    del SingleDiagAttClassifier
    torch.cuda.empty_cache()
    gc.collect()


    # Multi-head diagonalized attention mechanism (MultiDiagAtt)
    # ## Settings for MultiDiagAtt
    set_seed(42)
    tau = 1e3
    n_heads_alternative = [2, 3]
    path = os.path.join(checkpoint_path, 'MultiDiagAttClassifier.pt')
    training_params = {'batch_size': 64, 'epochs': 50, 'lr': 1e-2, 'opt': 'Adam'}  
    
    # ## Model selection for MultiDiagAtt
    CE_MultiDiagAtt, n_heads_MultiDiagAtt, MultiDiagAttClassifier = float("inf"), None, None
    MultiDiagAtt = MultiDiagAttRegression(x_dim=x_dim, tau=tau, n_heads=0)
    MultiDiagAtt.components.append(SingleDiagAtt)
    Classifier = RunRegression(model=MultiDiagAtt, task=task, patience=patience, delta=delta, path=path)
    for n_heads in n_heads_alternative:
        # For training
        Classifier.model.components.append(DiagAttRegression(x_dim, tau=tau).cuda("cuda"))
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
    
    # ## Evaluation for MultiDiagAtt
    Y_test_MultiDiagAttClassifier = MultiDiagAttClassifier.predict(X_test, batch_size=64)
    Accuracy_MultiDiagAtt.append(cal_accuracy(Y_test.ravel(), Y_test_MultiDiagAttClassifier.ravel()))

    # ## Representative tokens for MultiDiagAtt
    Tokens_MultiDiagAttClassifier = [localize_diagatt(MultiDiagAttClassifier.model.components[i], X, batch_size=64) for i in range(n_heads_MultiDiagAtt)]
    Tokens_MultiDiagAttClassifier = np.array(Tokens_MultiDiagAttClassifier).T
    Tokens_MultiDiagAttClassifier = Tokens[np.arange(len(Tokens))[:, None], Tokens_MultiDiagAttClassifier]
    Tokens_MultiDiagAtt.append(Tokens_MultiDiagAttClassifier)
            
    # ## Free memory for MultiDiagAtt
    del SingleDiagAtt, MultiDiagAtt, Classifier, MultiDiagAttClassifier
    torch.cuda.empty_cache()
    gc.collect()

end_time = time.time()
print(f"Total time: {end_time-start_time:.4f} | Average time: {(end_time-start_time)/n_repeats:.4f}")

# Summary
# ## Accuracy
Accuracy_mean = [np.mean(Accuracy_SingleDiagAtt), np.mean(Accuracy_MultiDiagAtt)]
Accuracy_std = [np.std(Accuracy_SingleDiagAtt), np.std(Accuracy_MultiDiagAtt)]
table = pd.DataFrame({"Accuracy_mean": Accuracy_mean, "Accuracy_std": Accuracy_std,})
table.index = ['1-head DiagAtt', 'R-head DiagAtt']
print(table.round(4))

# ## Tokens
# ### SingleDiagAtt
Positive_SingleDiagAtt, Negative_SingleDiagAtt = summarize(Tokens_SingleDiagAtt, Y, n_tops=10)
table = pd.DataFrame({'Positive_Tokens': list(Positive_SingleDiagAtt.keys()), 'Positive_Counters': list(Positive_SingleDiagAtt.values()),
                      'Negative_Tokens': list(Negative_SingleDiagAtt.keys()), 'Negative_Counters': list(Negative_SingleDiagAtt.values())})
print(table)

# ### MultiDiagAtt
Positive_MultiDiagAtt, Negative_MultiDiagAtt = summarize(Tokens_MultiDiagAtt, Y, n_tops=10)
table = pd.DataFrame({'Positive_Tokens': list(Positive_MultiDiagAtt.keys()), 'Positive_Counters': list(Positive_MultiDiagAtt.values()),
                      'Negative_Tokens': list(Negative_MultiDiagAtt.keys()), 'Negative_Counters': list(Negative_MultiDiagAtt.values())})
print(table)

