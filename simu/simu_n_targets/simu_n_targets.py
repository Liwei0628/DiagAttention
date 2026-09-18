import os 
import gc
import sys
import time
import copy
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
import torch

curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(curr_dir)
sys.path.append(os.path.join(root_dir, '..'))
from utils import cal_crsr, my_split, get_roi, get_DiagAttroi, get_StdAttcoef, set_seed, RegDataGenerator
from classMR import GroupLassoRegression, MDSPRegression, TwoStepRegression
from classMR import StdAttRegression, DiagAttRegression, MultiStdAttRegression, MultiDiagAttRegression, RunRegression

## task: reg
## n_samples: 27000
## x_dim: (100, 70)
## n_targets: s\in{2, 3}
## mu0s: [1, 4, -4]
## sigma0s: [3, 2, 2]
## rho: 0


# Basic settings
# ## Settings for data generating process
set_seed(42)                        # random seed
task = 'reg'                        # either regression('reg') or classification('cla')
n_samples = 27000                   # sample size for training
x_dim = (100, 70)                   # Shape of each X
rho = 0                             # correlation between non-target and target
mu0s_all = [1, 4, -4]               # a constant to control mu_target
sigma0s_all = [3, 2, 2]             # a constant to control sigma_target

# ## Settings for models and repetitions
n_tops = 10
delta = 1e-4
patience = 10
n_repeats = 100                                        # number of repeats
train_size, val_size, test_size = 0.7, 0.1, 0.2        # Train vs Validation vs Test
n_targets_alternative = [2, 3]                         # alternative numbers of target rows

# ## Path for checkpoints
checkpoint_path = os.path.join(curr_dir, 'checkpoints')
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

# ## Path for results
result_path = os.path.join(curr_dir, 'results')
if not os.path.exists(result_path):
    os.makedirs(result_path)

for n_targets in n_targets_alternative:
    # Lists for RMSE of different methods
    RMSE_GroupLasso, RMSE_MDSP, RMSE_TwoStep = [], [], []
    RMSE_SingleStdAtt, RMSE_MultiStdAtt, RMSE_SingleDiagAtt, RMSE_MultiDiagAtt = [], [], [], []

    # Lists for CRSR (correct row selection rate) of different methods
    CRSR_GroupLasso, CRSR_MDSP, CRSR_TwoStep = [], [], []
    CRSR_SingleStdAtt, CRSR_MultiStdAtt, CRSR_SingleDiagAtt, CRSR_MultiDiagAtt = [], [], [], []

    # Data generator for regression
    mu0s, sigma0s = mu0s_all[:n_targets], sigma0s_all[:n_targets]
    generator = RegDataGenerator(n_samples=int(n_samples/(train_size+val_size)), 
                                 x_dim=x_dim, n_targets=n_targets, rho=rho, mu0s=mu0s, sigma0s=sigma0s)

    start_time = time.time()
    for _ in range(n_repeats):
        # Generate data
        CR, X, Y = generator.gnrt_data()

        # Split data
        settings = {'train_size': train_size, 'val_size': val_size, 'test_size': test_size, 'shuffle': True}
        cr_train, cr_valid, CR_test, x_train, x_valid, X_test, y_train, y_valid, Y_test = my_split(CR, X, Y, **settings, random_state=42)
        print(f'For Y | S.D. | Train: {y_train.std():.3f}, Valid: {y_valid.std():.3f}, Test: {Y_test.std():.3f}')

        CR_train = np.concatenate([cr_train, cr_valid], axis=0)
        X_train, Y_train = np.concatenate([x_train, x_valid], axis=0), np.concatenate([y_train, y_valid], axis=0)


        # Group lasso regression (GroupLasso)
        # ## Settings for GroupLasso
        set_seed(42)
        penalty = 'group-lasso'
        alpha_alternative = [0.01, 0.1, 1]

        # ## Model selection for GroupLasso
        MSE_GroupLasso, alpha_GroupLasso, GroupLassoRegressor = float("inf"), None, None
        Regressor = GroupLassoRegression(x_dim=x_dim, task=task)
        for alpha in alpha_alternative:
            # For training
            training_params = {'alpha': alpha, 'max_itr': 200, 'tol': 1e-4}
            Regressor.fit(x_train, y_train, **training_params)

            # For model selection
            y_valid_hat = Regressor.predict(x_valid)
            MSE = root_mean_squared_error(y_valid.ravel(), y_valid_hat.ravel()) ** 2

            if MSE < MSE_GroupLasso:
                MSE_GroupLasso = MSE
                alpha_GroupLasso = alpha
                GroupLassoRegressor = copy.deepcopy(Regressor)
        
        # ## Training for GroupLasso
        training_params = {'alpha': alpha_GroupLasso, 'max_itr': 200, 'tol': 1e-4}
        GroupLassoRegressor.fit(X_train, Y_train, **training_params)

        # ## Evaluation for GroupLasso
        # ### For prediction performance
        Y_test_GroupLassoRegressor = GroupLassoRegressor.predict(X_test)
        RMSE_GroupLasso.append(root_mean_squared_error(Y_test.ravel(), Y_test_GroupLassoRegressor.ravel()))
        # ### For ROI recovery
        CR_test_GroupLassoRegressor = get_roi(GroupLassoRegressor.coef_[None, :, :], n_tops=n_tops)
        CR_test_GroupLassoRegressor = np.tile(CR_test_GroupLassoRegressor, (len(CR_test), 1))
        CRSR_GroupLasso.append(cal_crsr(CR_test, CR_test_GroupLassoRegressor))

        # ## Free memory for GroupLasso
        del Regressor, GroupLassoRegressor
        torch.cuda.empty_cache()
        gc.collect()


        # Multidirectional separation penalty (MDSP)
        # ## Settings for MDSP
        set_seed(42)
        penalty = "multidirectional"
        alpha_alternative = [0.01, 0.1, 1]

        # ## Model selection for MDSP
        MSE_MDSP, alpha_MDSP, MDSPRegressor = float("inf"), None, None
        Regressor = MDSPRegression(x_dim=x_dim, task=task)
        for alpha in alpha_alternative:
            # For training
            training_params = {'alpha': alpha, 'batch_size': 256, 'epochs': 50, 'lr': 1e-3, 'tol': 1e-4}  # For hyperparameter selection
            Regressor.train(train=(x_train, y_train), **training_params)

            # For model selection
            y_valid_hat, _ = Regressor.predict(test=(x_valid, y_valid), **training_params)
            MSE = root_mean_squared_error(y_valid.ravel(), y_valid_hat.ravel()) ** 2

            if MSE < MSE_MDSP:
                MSE_MDSP = MSE
                alpha_MDSP = alpha
                MDSPRegressor = copy.deepcopy(Regressor)
        
        # ## Training for MDSP
        training_params = {'alpha': alpha_MDSP, 'batch_size': 256, 'epochs': 50, 'lr': 1e-3, 'tol': 1e-4} 
        MDSPRegressor.train(train=(X_train, Y_train), **training_params)

        # ## Evaluation for MDSP
        # ### For prediction performance
        Y_test_MDSPRegressor, Cs_test_MDSPRegressor = MDSPRegressor.predict(test=(X_test, Y_test), **training_params)
        RMSE_MDSP.append(root_mean_squared_error(Y_test.ravel(), Y_test_MDSPRegressor.ravel()))
        # ### For ROI recovery
        CR_test_MDSPRegressor = get_roi(Cs_test_MDSPRegressor, n_tops=n_tops, batch_size=256)
        CRSR_MDSP.append(cal_crsr(CR_test, CR_test_MDSPRegressor))

        # ## Free memory for MDSP
        del Regressor, MDSPRegressor, Cs_test_MDSPRegressor
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
        Regressor = TwoStepRegression(x_dim=x_dim, task=task, n_tops=n_tops, n_components=n_components)
        Regressor.fit_gmm(X_train, **training_params)

        # ## Model selection for Ridge
        MSE_TwoStep, alpha_TwoStep, TwoStepRegressor = float("inf"), None, None
        for k in range(n_components):
            Regressor.k_best = k
            for alpha in alpha_alternative:
                # For training
                training_params = {'alpha': alpha, 'max_itr': 200, 'tol': 1e-4}
                Regressor.fit_reg(x_train, y_train, **training_params)

                # For model selection
                _, y_valid_hat = Regressor.predict(x_valid)
                MSE = root_mean_squared_error(y_valid.ravel(), y_valid_hat.ravel()) ** 2

                if MSE < MSE_TwoStep:
                    MSE_TwoStep = MSE
                    alpha_TwoStep = alpha
                    TwoStepRegressor = copy.deepcopy(Regressor)
        
        # ## Training for Ridge
        training_params = {'alpha': alpha_TwoStep, 'max_itr': 200, 'tol': 1e-4}
        TwoStepRegressor.fit_reg(X_train, Y_train, **training_params)

        # ## Evaluation for TwoStep
        # ### For prediction performance
        CR_test_TwoStepRegressor, Y_test_TwoStepRegressor = TwoStepRegressor.predict(X_test)
        RMSE_TwoStep.append(root_mean_squared_error(Y_test.ravel(), Y_test_TwoStepRegressor.ravel()))
        # ### For ROI recovery
        CRSR_TwoStep.append(cal_crsr(CR_test, CR_test_TwoStepRegressor))

        # ## Free memory for TwoStep
        del Regressor, TwoStepRegressor
        torch.cuda.empty_cache()
        gc.collect()


        # Single-head diagonalized attention mechanism (SingleDiagAtt)
        # ## Settings for SingleDiagAtt
        set_seed(42)
        lam = 1e-2
        penalty = None
        path = os.path.join(checkpoint_path, 'SingleDiagAttRegressor.pt')
        training_params = {'batch_size': 64, 'epochs': 50, 'lr': 1e-2, 'opt': 'Adam'}  
        
        # ## Training for SingleDiagAtt
        SingleDiagAtt = DiagAttRegression(x_dim=x_dim, lam=lam)
        SingleDiagAttRegressor = RunRegression(model=SingleDiagAtt, task=task, patience=patience, delta=delta, path=path)
        SingleDiagAttRegressor.train(train=(X_train, Y_train), valid=(X_train, Y_train), **training_params)
        W = SingleDiagAttRegressor.model.W.detach().cpu().numpy()

        # ## Evaluation for SingleDiagAtt
        # ### For prediction performance
        Y_test_SingleDiagAttRegressor = SingleDiagAttRegressor.predict(X_test, batch_size=64)
        RMSE_SingleDiagAtt.append(root_mean_squared_error(Y_test.ravel(), Y_test_SingleDiagAttRegressor.ravel()))
        # ### For ROI recovery
        CR_test_SingleDiagAttRegressor = get_DiagAttroi(X_test, W, batch_size=64).reshape(-1, 1)
        CRSR_SingleDiagAtt.append(cal_crsr(CR_test, CR_test_SingleDiagAttRegressor))
        
        # ## Free memory for SingleDiagAtt
        del W, SingleDiagAttRegressor
        torch.cuda.empty_cache()
        gc.collect()


        # Multi-head diagonalized attention mechanism (MultiDiagAtt)
        # ## Settings for MultiDiagAtt
        set_seed(42)
        lam = 1e-2
        penalty = None
        n_heads_alternative = list(range(2, n_targets + 3, 1))
        path = os.path.join(checkpoint_path, 'MultiDiagAttRegressor.pt')
        training_params = {'batch_size': 64, 'epochs': 50, 'lr': 1e-2, 'opt': 'Adam'}  

        # ## Model selection for MultiDiagAtt
        MSE_MultiDiagAtt, n_heads_MultiDiagAtt, MultiDiagAttRegressor = float("inf"), None, None
        MultiDiagAtt = MultiDiagAttRegression(x_dim=x_dim, lam=lam, n_heads=0)
        MultiDiagAtt.attentions.append(SingleDiagAtt)
        Regressor = RunRegression(model=MultiDiagAtt, task=task, patience=patience, delta=delta, path=path)
        for n_heads in n_heads_alternative:
            # For training
            Regressor.model.attentions.append(DiagAttRegression(x_dim, lam=lam).cuda("cuda:1"))
            Regressor.train(train=(x_train, y_train), valid=(x_train, y_train), **training_params)

            # For model selection
            y_valid_hat = Regressor.predict(x_valid, batch_size=64)
            MSE = root_mean_squared_error(y_valid.ravel(), y_valid_hat.ravel()) ** 2

            if MSE < MSE_MultiDiagAtt:
                MSE_MultiDiagAtt = MSE
                n_heads_MultiDiagAtt = n_heads
                MultiDiagAttRegressor = copy.deepcopy(Regressor)
        
        # ## Training for MultiDiagAtt
        MultiDiagAttRegressor.train(train=(X_train, Y_train), valid=(X_train, Y_train), **training_params)
        Ws = [MultiDiagAttRegressor.model.attentions[i].W.detach().cpu().numpy() for i in range(n_heads_MultiDiagAtt)]

        # ## Evaluation for MultiDiagAtt
        # ### For prediction performance
        Y_test_MultiDiagAttRegressor = MultiDiagAttRegressor.predict(X_test, batch_size=64)
        RMSE_MultiDiagAtt.append(root_mean_squared_error(Y_test.ravel(), Y_test_MultiDiagAttRegressor.ravel()))
        # ### For ROI recovery
        CR_test_MultiDiagAttRegressor = [get_DiagAttroi(X_test, Ws[i], batch_size=64) for i in range(n_heads_MultiDiagAtt)]
        CR_test_MultiDiagAttRegressor = np.array(CR_test_MultiDiagAttRegressor).T
        CRSR_MultiDiagAtt.append(cal_crsr(CR_test, CR_test_MultiDiagAttRegressor))
        
        # ## Free memory for MultiDiagAtt
        del Ws, SingleDiagAtt, MultiDiagAtt, Regressor, MultiDiagAttRegressor
        torch.cuda.empty_cache()
        gc.collect()


        # Single-head standard attention mechanism (SingleStdAtt)
        # ## Settings for SingleStdAtt
        set_seed(42)
        penalty = None
        path = os.path.join(checkpoint_path, 'SingleStdAttRegressor.pt')
        training_params = {'batch_size': 512, 'epochs': 50, 'lr': 1e-4, 'opt': 'Adam'}
        
        # ## Training for SingleStdAtt
        SingleStdAtt = StdAttRegression(x_dim=x_dim)
        SingleStdAttRegressor = RunRegression(model=SingleStdAtt, task=task, patience=patience, delta=delta, path=path)
        SingleStdAttRegressor.train(train=(X_train, Y_train), valid=(X_train, Y_train), **training_params)
        
        # ## Evaluation for SingleStdAtt
        # ### For prediction performance
        Y_test_SingleStdAttRegressor = SingleStdAttRegressor.predict(X_test, batch_size=64)
        RMSE_SingleStdAtt.append(root_mean_squared_error(Y_test.ravel(), Y_test_SingleStdAttRegressor.ravel()))
        # ### For ROI recovery
        Cs_test_SingleStdAttRegressor = get_StdAttcoef(X_test, SingleStdAttRegressor.model, batch_size=64)
        CR_test_SingleStdAttRegressor = get_roi(Cs_test_SingleStdAttRegressor, n_tops=n_tops, batch_size=64)
        CRSR_SingleStdAtt.append(cal_crsr(CR_test, CR_test_SingleStdAttRegressor))
        
        # ## Free memory for SingleStdAtt
        del SingleStdAttRegressor, Cs_test_SingleStdAttRegressor
        torch.cuda.empty_cache()
        gc.collect()


        # Multi-head standard attention mechanism (MultiStdAtt)
        # ## Settings for MultiStdAtt
        set_seed(42)
        penalty = None
        path = os.path.join(checkpoint_path, 'MultiStdAttRegressor.pt')
        training_params = {'batch_size': 512, 'epochs': 50, 'lr': 1e-4, 'opt': 'Adam'}

        # ## Training for MultiStdAtt
        MultiStdAtt = MultiStdAttRegression(x_dim=x_dim, n_heads=n_heads_MultiDiagAtt)
        MultiStdAtt.attentions[0] = SingleStdAtt
        MultiStdAttRegressor = RunRegression(model=MultiStdAtt, task=task, patience=patience, delta=delta, path=path)
        MultiStdAttRegressor.train(train=(X_train, Y_train), valid=(X_train, Y_train), **training_params)

        # ## Evaluation for MultiStdAtt
        # ### For prediction performance
        Y_test_MultiStdAttRegressor = MultiStdAttRegressor.predict(X_test, batch_size=64)
        RMSE_MultiStdAtt.append(root_mean_squared_error(Y_test.ravel(), Y_test_MultiStdAttRegressor.ravel()))
        # ### For ROI recovery
        Cs_test_MultiStdAttRegressor = [get_StdAttcoef(X_test, attention, batch_size=64) for attention in MultiStdAttRegressor.model.attentions]
        CR_test_MultiStdAttRegressor = [get_roi(Cs_test, n_tops=n_tops//n_heads_MultiDiagAtt, batch_size=64) for Cs_test in Cs_test_MultiStdAttRegressor]
        CR_test_MultiStdAttRegressor = np.concatenate(CR_test_MultiStdAttRegressor, axis=1)
        CRSR_MultiStdAtt.append(cal_crsr(CR_test, CR_test_MultiStdAttRegressor))

        # ## Free memory for MultiStdAtt
        del SingleStdAtt, MultiStdAtt, MultiStdAttRegressor, Cs_test_MultiStdAttRegressor
        torch.cuda.empty_cache()
        gc.collect()

    end_time = time.time()
    print(f"Cardinality: {n_targets} | Total time: {end_time-start_time:.4f} | Average time: {(end_time-start_time)/n_repeats:.4f}")

    # Summary
    # ## RMSE
    table = pd.DataFrame({'GroupLasso': RMSE_GroupLasso, 'MDSP': RMSE_MDSP, 'Two-Step': RMSE_TwoStep,
                          'SingleStdAtt': RMSE_SingleStdAtt, 'MultiStdAtt': RMSE_MultiStdAtt, 'SingleDiagAtt': RMSE_SingleDiagAtt, 'MultiDiagAtt': RMSE_MultiDiagAtt})
    table.to_csv(os.path.join(result_path, f"performance-RMSE (n_targets={n_targets}).csv"), index=False)

    # ## CRSR
    table = pd.DataFrame({'GroupLasso': CRSR_GroupLasso, 'MDSP': CRSR_MDSP, 'Two-Step': CRSR_TwoStep,
                          'SingleStdAtt': CRSR_SingleStdAtt, 'MultiStdAtt': CRSR_MultiStdAtt, 'SingleDiagAtt': CRSR_SingleDiagAtt, 'MultiDiagAtt': CRSR_MultiDiagAtt})
    table.to_csv(os.path.join(result_path, f"performance-CRSR (n_targets={n_targets}).csv"), index=False)

    # ## Mean & S.D.
    RMSE_mean = [np.mean(RMSE_GroupLasso), np.mean(RMSE_MDSP), np.mean(RMSE_TwoStep), 
                 np.mean(RMSE_SingleStdAtt), np.mean(RMSE_MultiStdAtt), np.mean(RMSE_SingleDiagAtt), np.mean(RMSE_MultiDiagAtt)]
    RMSE_std = [np.std(RMSE_GroupLasso), np.std(RMSE_MDSP), np.std(RMSE_TwoStep), 
                np.std(RMSE_SingleStdAtt), np.std(RMSE_MultiStdAtt), np.std(RMSE_SingleDiagAtt), np.std(RMSE_MultiDiagAtt)]
    
    CRSR_mean = [np.mean(CRSR_GroupLasso), np.mean(CRSR_MDSP), np.mean(CRSR_TwoStep), 
                 np.mean(CRSR_SingleStdAtt), np.mean(CRSR_MultiStdAtt), np.mean(CRSR_SingleDiagAtt), np.mean(CRSR_MultiDiagAtt)]
    CRSR_std = [np.std(CRSR_GroupLasso), np.std(CRSR_MDSP), np.std(CRSR_TwoStep), 
                np.std(CRSR_SingleStdAtt), np.std(CRSR_MultiStdAtt), np.std(CRSR_SingleDiagAtt), np.std(CRSR_MultiDiagAtt)]

    table = pd.DataFrame({"RMSE_mean": RMSE_mean, "RMSE_std": RMSE_std, "CRSR_mean": CRSR_mean, "CRSR_std": CRSR_std,})
    table.index = ['GroupLasso', 'MDSP', 'Two-Step', 'SingleStdAtt', 'MultiStdAtt', 'SingleDiagAtt', 'MultiDiagAtt']
    print(table.round(4))
    table.to_csv(os.path.join(result_path, f"performance-All (n_targets={n_targets}).csv"), index=True)

