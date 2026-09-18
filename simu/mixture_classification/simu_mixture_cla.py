import os 
import gc
import sys
import time
import copy
from itertools import product
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
import torch

curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(curr_dir)
sys.path.append(os.path.join(root_dir, '..'))
from utils import cal_accuracy, cal_crsr, my_split, get_roi, get_DiagAttroi, get_StdAttcoef, set_seed, ClaDataGenerator
from classMR import GroupLassoRegression, MDSPRegression, TwoStepRegression
from classMR import StdAttRegression, DiagAttRegression, MultiStdAttRegression, MultiDiagAttRegression, RunRegression

## task: cla
## n_samples: n\in{9000, 18000}
## x_dim: (p,d)\in{(100, 70), (100, 90), (120, 70), (120, 90)}
## n_targets: 1
## mu0s: mu0_pos=3, mu0_neg=-3
## sigma0s: sigma_pos=sigma_neg=2
## rho: 0


# Basic settings
# ## Settings for data generating process
set_seed(42)                 # random seed
task = 'cla'                 # classification('cla')
pi = 0.5                     # probability to control positive and negative
rho = 0                      # correlation level
mu0s = [3, -3]               # constants to control mu0_pos and mu0_neg
sigma0s = [2, 2]             # constants to control sigma0_pos and sigma0_neg

# ## Settings for models and repetitions
n_tops = 10
delta = 1e-4
patience = 10
n_repeats = 100                                                        # number of repeats
train_size, val_size, test_size = 0.7, 0.1, 0.2                        # Train vs Validation vs Test
n_samples_alternative = [9000, 18000]                                  # alternative sample sizes for training
x_dim_alternative  = [(100, 70), (100, 90), (120, 70), (120, 90)]      # alternative matrix dimensions

# ## Path for checkpoints
checkpoint_path = os.path.join(curr_dir, 'checkpoints')
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

# ## Path for results
result_path = os.path.join(curr_dir, 'results')
if not os.path.exists(result_path):
    os.makedirs(result_path)


for (n_samples, x_dim) in list(product(n_samples_alternative, x_dim_alternative)):
    # Lists for Accuracy of different methods
    Accuracy_GroupLasso, Accuracy_MDSP, Accuracy_TwoStep = [], [], []
    Accuracy_SingleStdAtt, Accuracy_MultiStdAtt, Accuracy_SingleDiagAtt, Accuracy_MultiDiagAtt = [], [], [], []

    # Lists for CRSR (correct row selection rate) of different methods
    CRSR_GroupLasso, CRSR_MDSP, CRSR_TwoStep = [], [], []
    CRSR_SingleStdAtt, CRSR_MultiStdAtt, CRSR_SingleDiagAtt, CRSR_MultiDiagAtt = [], [], [], []

    # Data generator for classification
    generator = ClaDataGenerator(n_samples=int(n_samples/(train_size+val_size)),
                                 x_dim=x_dim, pi=pi, mu0s=mu0s, sigma0s=sigma0s, rho=0)


    start_time = time.time()
    for _ in range(n_repeats):
        # Generate data
        CR, X, Y = generator.gnrt_data()

        # Split data
        settings = {'train_size': train_size, 'val_size': val_size, 'test_size': test_size, 'shuffle': True}
        cr_train, cr_valid, CR_test, x_train, x_valid, X_test, y_train, y_valid, Y_test = my_split(CR, X, Y, **settings, random_state=42)
        print(f'For Y | Positive | Train: {y_train.mean():.3f}, Valid: {y_valid.mean():.3f}, Test: {Y_test.mean():.3f}')

        CR_train = np.concatenate([cr_train, cr_valid], axis=0)
        X_train, Y_train = np.concatenate([x_train, x_valid], axis=0), np.concatenate([y_train, y_valid], axis=0)


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
            training_params = {'alpha': alpha, 'max_itr': 200, 'tol': 1e-4}
            Classifier.fit(x_train, y_train, **training_params)

            # For model selection
            y_valid_hat = Classifier.predict(x_valid)
            CE = log_loss(y_valid.ravel(), y_valid_hat.ravel(), normalize=False)

            if CE < CE_GroupLasso:
                CE_GroupLasso = CE
                alpha_GroupLasso = alpha
                GroupLassoClassifier = copy.deepcopy(Classifier)

        # ## Training for GroupLasso
        training_params = {'alpha': alpha_GroupLasso, 'max_itr': 200, 'tol': 1e-4}
        GroupLassoClassifier.fit(X_train, Y_train, **training_params)

        # ## Evaluation for GroupLasso
        # ### For prediction performance
        Y_test_GroupLassoClassifier = GroupLassoClassifier.predict(X_test)
        Accuracy_GroupLasso.append(cal_accuracy(Y_test.ravel(), Y_test_GroupLassoClassifier.ravel()))
        # ### For ROI identification accuracy
        CR_test_GroupLassoClassifier = get_roi(GroupLassoClassifier.coef_[None, :, :], n_tops=n_tops)
        CR_test_GroupLassoClassifier = np.tile(CR_test_GroupLassoClassifier, (len(CR_test), 1))
        CRSR_GroupLasso.append(cal_crsr(CR_test, CR_test_GroupLassoClassifier))

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
        CE_MDSP, alpha_MDSP, MDSPClassifier = float("inf"), None, None
        Classifier = MDSPRegression(x_dim=x_dim, task=task)
        for alpha in alpha_alternative:
            # For training
            training_params = {'alpha': alpha, 'batch_size': 64, 'epochs': 50, 'lr': 5e-5, 'tol': 1e-4}  # For hyperparameter selection
            Classifier.train(train=(x_train, y_train), **training_params)

            # For model selection
            y_valid_hat, _ = Classifier.predict(test=(x_valid, y_valid), **training_params)
            CE = log_loss(y_valid.ravel(), y_valid_hat.ravel(), normalize=False)

            if CE < CE_MDSP:
                CE_MDSP = CE
                alpha_MDSP = alpha
                MDSPClassifier = copy.deepcopy(Classifier)
        
        # ## Training for MDSP
        training_params = {'alpha': alpha_MDSP, 'batch_size': 64, 'epochs': 50, 'lr': 5e-5, 'tol': 1e-4}
        MDSPClassifier.train(train=(X_train, Y_train), **training_params)

        # ## Evaluation for MDSP
        # ### For prediction performance
        Y_test_MDSPClassifier, Cs_test_MDSPClassifier = MDSPClassifier.predict(test=(X_test, Y_test), **training_params)
        Accuracy_MDSP.append(cal_accuracy(Y_test.ravel(), Y_test_MDSPClassifier.ravel()))
        # ### For ROI recovery
        CR_test_MDSPClassifier = get_roi(Cs_test_MDSPClassifier, n_tops=n_tops, batch_size=256)
        CRSR_MDSP.append(cal_crsr(CR_test, CR_test_MDSPClassifier))

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

        # ## Training for GMM
        training_params = {'covariance_type': 'full', 'max_itr': 200, 'tol': 1e-2}
        Classifier = TwoStepRegression(x_dim=x_dim, task=task, n_tops=n_tops, n_components=n_components)
        Classifier.fit_gmm(X_train, **training_params)

        # ## Model selection for Ridge
        CE_TwoStep, alpha_TwoStep, TwoStepClassifier = float("inf"), None, None
        for k in range(n_components):
            Classifier.k_best = k
            for alpha in alpha_alternative:
                # For training
                training_params = {'alpha': alpha, 'max_itr': 200, 'tol': 1e-4}
                Classifier.fit_reg(x_train, y_train, **training_params)

                # For model selection
                _, y_valid_hat = Classifier.predict(x_valid)
                CE = log_loss(y_valid.ravel(), y_valid_hat.ravel(), normalize=False)

                if CE < CE_TwoStep:
                    CE_TwoStep = CE
                    alpha_TwoStep = alpha
                    TwoStepClassifier = copy.deepcopy(Classifier)
        
        # ## Training for Ridge
        training_params = {'alpha': alpha_TwoStep, 'max_itr': 200, 'tol': 1e-4}
        TwoStepClassifier.fit_reg(X_train, Y_train, **training_params)

        # ## Evaluation for TwoStep
        # ### For prediction performance
        CR_test_TwoStepClassifier, Y_test_TwoStepClassifier = TwoStepClassifier.predict(X_test)
        Accuracy_TwoStep.append(cal_accuracy(Y_test.ravel(), Y_test_TwoStepClassifier.ravel()))
        # ### For ROI recovery
        CRSR_TwoStep.append(cal_crsr(CR_test, CR_test_TwoStepClassifier))

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
        # ### For prediction performance
        Y_test_SingleDiagAttClassifier = SingleDiagAttClassifier.predict(X_test, batch_size=64)
        Accuracy_SingleDiagAtt.append(cal_accuracy(Y_test.ravel(), Y_test_SingleDiagAttClassifier.ravel()))
        # ### For ROI recovery
        CR_test_SingleDiagAttClassifier = get_DiagAttroi(X_test, W, batch_size=64).reshape(-1, 1)
        CRSR_SingleDiagAtt.append(cal_crsr(CR_test, CR_test_SingleDiagAttClassifier))
        
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
        # ### For prediction performance
        Y_test_MultiDiagAttClassifier = MultiDiagAttClassifier.predict(X_test, batch_size=64)
        Accuracy_MultiDiagAtt.append(cal_accuracy(Y_test.ravel(), Y_test_MultiDiagAttClassifier.ravel()))
        # ### For ROI recovery
        CR_test_MultiDiagAttClassifier = [get_DiagAttroi(X_test, Ws[i], batch_size=64) for i in range(n_heads_MultiDiagAtt)]
        CR_test_MultiDiagAttClassifier = np.array(CR_test_MultiDiagAttClassifier).T
        CRSR_MultiDiagAtt.append(cal_crsr(CR_test, CR_test_MultiDiagAttClassifier))
        
        # ## Free memory for MultiDiagAtt
        del Ws, SingleDiagAtt, MultiDiagAtt, Classifier, MultiDiagAttClassifier
        torch.cuda.empty_cache()
        gc.collect()


        # Single-head standard attention mechanism (SingleStdAtt)
        # ## Settings for SingleStdAtt
        set_seed(42)
        penalty = None
        path = os.path.join(checkpoint_path, 'SingleStdAttClassifier.pt')
        training_params = {'batch_size': 64, 'epochs': 50, 'lr': 1e-4, 'opt': 'Adam'}
        
        # ## Training for SingleStdAtt
        SingleStdAtt = StdAttRegression(x_dim=x_dim)
        SingleStdAttClassifier = RunRegression(model=SingleStdAtt, task=task, patience=patience, delta=delta, path=path)
        SingleStdAttClassifier.train(train=(X_train, Y_train), valid=(X_train, Y_train), **training_params)
        
        # ## Evaluation for SingleStdAtt
        # ### For prediction performance
        Y_test_SingleStdAttClassifier = SingleStdAttClassifier.predict(X_test, batch_size=64)
        Accuracy_SingleStdAtt.append(cal_accuracy(Y_test.ravel(), Y_test_SingleStdAttClassifier.ravel()))
        # ### For ROI recovery
        Cs_test_SingleStdAttClassifier = get_StdAttcoef(X_test, SingleStdAttClassifier.model, batch_size=64)
        CR_test_SingleStdAttClassifier = get_roi(Cs_test_SingleStdAttClassifier, n_tops=n_tops, batch_size=64)
        CRSR_SingleStdAtt.append(cal_crsr(CR_test, CR_test_SingleStdAttClassifier))
        
        # ## Free memory for SingleStdAtt
        del SingleStdAttClassifier, Cs_test_SingleStdAttClassifier
        torch.cuda.empty_cache()
        gc.collect()


        # Multi-head standard attention mechanism (MultiStdAtt)
        # ## Settings for MultiStdAtt
        set_seed(42)
        penalty = None
        path = os.path.join(checkpoint_path, 'MultiStdAttClassifier.pt')
        training_params = {'batch_size': 64, 'epochs': 50, 'lr': 1e-4, 'opt': 'Adam'}

        # ## Training for MultiStdAtt
        MultiStdAtt = MultiStdAttRegression(x_dim=x_dim, n_heads=n_heads_MultiDiagAtt)
        MultiStdAtt.attentions[0] = SingleStdAtt
        MultiStdAttClassifier = RunRegression(model=MultiStdAtt, task=task, patience=patience, delta=delta, path=path)
        MultiStdAttClassifier.train(train=(X_train, Y_train), valid=(X_train, Y_train), **training_params)

        # ## Evaluation for MultiStdAtt
        # ### For prediction performance
        Y_test_MultiStdAttClassifier = MultiStdAttClassifier.predict(X_test, batch_size=64)
        Accuracy_MultiStdAtt.append(cal_accuracy(Y_test.ravel(), Y_test_MultiStdAttClassifier.ravel()))
        # ### For ROI recovery
        Cs_test_MultiStdAttClassifier = [get_StdAttcoef(X_test, attention, batch_size=64) for attention in MultiStdAttClassifier.model.attentions]
        CR_test_MultiStdAttClassifier = [get_roi(Cs_test, n_tops=n_tops//n_heads_MultiDiagAtt, batch_size=64) for Cs_test in Cs_test_MultiStdAttClassifier]
        CR_test_MultiStdAttClassifier = np.concatenate(CR_test_MultiStdAttClassifier, axis=1)
        CRSR_MultiStdAtt.append(cal_crsr(CR_test, CR_test_MultiStdAttClassifier))

        # ## Free memory for MultiStdAtt
        del SingleStdAtt, MultiStdAtt, MultiStdAttClassifier, Cs_test_MultiStdAttClassifier
        torch.cuda.empty_cache()
        gc.collect()

    end_time = time.time()
    print(f"Sample size: {n_samples} | Dimension: {x_dim} | Total time: {end_time-start_time:.4f} | Average time: {(end_time-start_time)/n_repeats:.4f}")

    # Summary
    # ## Accuracy
    table = pd.DataFrame({'GroupLasso': Accuracy_GroupLasso, 'MDSP': Accuracy_MDSP, 'Two-Step': Accuracy_TwoStep,
                          'SingleStdAtt': Accuracy_SingleStdAtt, 'MultiStdAtt': Accuracy_MultiStdAtt, 'SingleDiagAtt': Accuracy_SingleDiagAtt, 'MultiDiagAtt': Accuracy_MultiDiagAtt})
    table.to_csv(os.path.join(result_path, f"performance-Accuracy (n_samples={n_samples}, x_dim={x_dim}).csv"), index=False)

    # ## CRSR
    table = pd.DataFrame({'GroupLasso': CRSR_GroupLasso, 'MDSP': CRSR_MDSP, 'Two-Step': CRSR_TwoStep,
                          'SingleStdAtt': CRSR_SingleStdAtt, 'MultiStdAtt': CRSR_MultiStdAtt, 'SingleDiagAtt': CRSR_SingleDiagAtt, 'MultiDiagAtt': CRSR_MultiDiagAtt})
    table.to_csv(os.path.join(result_path, f"performance-CRSR (n_samples={n_samples}, x_dim={x_dim}).csv"), index=False)

    # ## Mean & S.D.
    Accuracy_mean = [np.mean(Accuracy_GroupLasso), np.mean(Accuracy_MDSP), np.mean(Accuracy_TwoStep), 
                     np.mean(Accuracy_SingleStdAtt), np.mean(Accuracy_MultiStdAtt), np.mean(Accuracy_SingleDiagAtt), np.mean(Accuracy_MultiDiagAtt)]
    Accuracy_std = [np.std(Accuracy_GroupLasso), np.std(Accuracy_MDSP), np.std(Accuracy_TwoStep), 
                    np.std(Accuracy_SingleStdAtt), np.std(Accuracy_MultiStdAtt), np.std(Accuracy_SingleDiagAtt), np.std(Accuracy_MultiDiagAtt)]
    
    CRSR_mean = [np.mean(CRSR_GroupLasso), np.mean(CRSR_MDSP), np.mean(CRSR_TwoStep), 
                 np.mean(CRSR_SingleStdAtt), np.mean(CRSR_MultiStdAtt), np.mean(CRSR_SingleDiagAtt), np.mean(CRSR_MultiDiagAtt)]
    CRSR_std = [np.std(CRSR_GroupLasso), np.std(CRSR_MDSP), np.std(CRSR_TwoStep), 
                np.std(CRSR_SingleStdAtt), np.std(CRSR_MultiStdAtt), np.std(CRSR_SingleDiagAtt), np.std(CRSR_MultiDiagAtt)]

    table = pd.DataFrame({"Accuracy_mean": Accuracy_mean, "Accuracy_std": Accuracy_std, "CRSR_mean": CRSR_mean, "CRSR_std": CRSR_std,})
    table.index = ['GroupLasso', 'MDSP', 'Two-Step', 'SingleStdAtt', 'MultiStdAtt', 'SingleDiagAtt', 'MultiDiagAtt']
    print(table.round(4))
    table.to_csv(os.path.join(result_path, f"performance-All (n_samples={n_samples}, x_dim={x_dim}).csv"), index=True)

