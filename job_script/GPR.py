import numpy as np
import pandas as pd
import tensorflow as tf
import math
import gpflow
import warnings
from sklearn.neighbors import NearestNeighbors
from scipy.stats import qmc, norm

def normalize(inputarray, skScaler, method = 'Standardization'):
    '''
    inputarray = data to be scaled
    x_test_g = test data to be used in calculating mean and standard deviation to standardize data
    skScaler = test data 
    Method = Standardization method which could be 
    - Standardization: directly using the mean and standard deviation to scale the data
    - MinMaxNorm - scales the data to range -1 and 1 where minimum data is scaled to -1 and max is scaled to 1
    - Log: scaling using log10 of the data
    - LogStandardization: data is first scaled logarithmically before standardization
    '''
    ## check for method
    if method == 'Standardization': 
        aux = inputarray
    elif method == 'LogStandardization' or method == 'Log':
        aux = np.log10(inputarray)
        skScaler = np.log10(skScaler)
    elif method == 'MinMaxNorm':
        aux = inputarray
    else:
        raise ValueError('Could not recognize method provided')
    if method == 'LogStandardization' or method == 'Standardization':
        x_mean = np.mean(skScaler)
        x_std = np.std(skScaler)
        outputarray = (aux - x_mean) / x_std
    elif method == 'MinMaxNorm':
        min_data = np.min(skScaler)
        max_data = np.max(skScaler)
        outputarray = 2 * ((aux - min_data) / (max_data - min_data)) - 1
    else:
        outputarray = aux
    return outputarray

def reverse(inputarray, skScaler, method = 'Standardization'):
    # to rescale the data back to its original distribution
    if method == 'LogStandardization' or method == 'Log':
        skScaler = np.log10(skScaler)
    if method == 'Standardization' or method == 'LogStandardization':
        x_mean = np.mean(skScaler)
        x_std = np.std(skScaler)
        outputarray = (inputarray * x_std) + x_mean
    elif method == 'MinMaxNorm':
        min_data = np.min(skScaler)
        max_data = np.max(skScaler)
        outputarray = ((inputarray + 1)/2) * (max_data - min_data) + min_data
    else: 
        outputarray = 10 ** inputarray
    if method == 'LogStandardization':
        outputarray = 10 ** outputarray
    
    return outputarray

def round_up(x):
    new_x = round(abs(x))
    
    if new_x == 0:
        new_x = 1
    
    length = str(new_x)
    length = len(length)

    round_value = length - 1
    rounded = round(new_x, -round_value)
    rounded = round(rounded)

    return rounded

def gp(x, y, config = {}, mean_func = None):
    # unpack the configuration dictionary
    tf.compat.v1.enable_eager_execution()
    kernel = config.get('kernel', 'RQ')
    usewhitekernel = config.get('useWhiteKernel', 'True')
    trainLikelihood = config.get('trainLikelihood', 'True')

    # select and initialize the kernels
    if kernel == 'RBF':
        gp_kernel = gpflow.kernels.SquaredExponential()
    if kernel == 'RQ':
        gp_kernel = gpflow.kernels.RationalQuadratic()
    if kernel == 'Matern12':
        gp_kernel = gpflow.kernels.Matern12()
    if kernel == 'Matern52':
        gp_kernel = gpflow.kernels.Matern52()

    # Add white kernel
    if usewhitekernel:
        gp_kernel = gp_kernel + gpflow.kernels.White()

    # build gp model

    model = gpflow.models.GPR((x, y), gp_kernel, mean_function = mean_func, noise_variance = 10 ** -2)

    # select whether likelihood will be treated as a trainable parameter or not
    
    gpflow.utilities.set_trainable(model.likelihood.variance,trainLikelihood)

    # build optimizer
    
    opt = gpflow.optimizers.Scipy()

    # fit GP to training data
    
    aux = opt.minimize(model.training_loss, model.trainable_variables, options = dict(maxiter = 2000, ftol = 1e-9), method = 'l-bfgs-b', step_callback = None, compile = True, allow_unused_variables = False)

    # check convergence
    if aux.success == False:
        warnings.warn('GP optimizer failed to converge')
    
    #output
    
    return model

def gp_predict(model, x):

    # perform gp prediction and get mean and variance

    GP_mean, GP_var = model.predict_f(x)

    # convert to numpy array

    # GP_mean = GP_mean.numpy()
    # GP_var = GP_var.numpy()

    # prepare output

    y_pred = GP_mean
    sigma = GP_var ** 0.5

    # output

    return y_pred, sigma

def data_padding(X, dim):
    '''
    Pads input vectors with zeros up to specified dimensions

    X: input array with varying length and dimension
    dim: desired dimensions for padded data
    '''
    pad = np.zeros((len(X), dim))
    for i, x in enumerate(X):
        pad[i, :len(x)] = x
    return pad

def lhs_sampler(X, n_samples, seed = 42):

    # define LHS sampler
    sampler = qmc.LatinHypercube(d = 1, seed= seed)
    # Generate LHS samples
    lhs_unit = sampler.random(n = n_samples)

    # define upper and lower bounds for scaling the lhs units
    l_bound = X.min(axis=0)
    u_bound = X.max(axis=0)
    lhs_scaled = qmc.scale(lhs_unit, l_bound, u_bound)

    # Fit Nearest Neighbors - for each LHS point, find the nearest neighbors in the dataset
    NN = NearestNeighbors(n_neighbors=1).fit(X)
    _, idx = NN.kneighbors(lhs_scaled)

    # Get unique indices to avoid repetitions
    unique_idx = np.unique(idx)

    return unique_idx

def train_data_generator(dataframe, no_samples, seed, pressure_index, uptake_index, error_index, mol_frac_index = None, Temp_index = None):
    # Define actual pressure data
    X = dataframe.iloc[:, pressure_index]
    X = X.dropna()
    X = np.atleast_2d(X).flatten().reshape(-1,1)

    sample_counter = 0

    while sample_counter < no_samples:
        # Extract data at the defined index
        # Pressure data
        X_train = X[lhs_sampler(X, no_samples, seed)]
        X_train = np.atleast_2d(X_train).flatten().reshape(-1,1)
        # print(X_train.shape)

        # Mole fraction data
        if mol_frac_index != None:
            X2 = dataframe.iloc[:, mol_frac_index]
            X2_train = X2[lhs_sampler(X, no_samples, seed)]
            X2_train = np.atleast_2d(X2_train).flatten().reshape(-1,1)
            # print(X2_train.shape)

        # Temperature data
        if Temp_index != None:
            T = dataframe.iloc[:, Temp_index]
            T_train = T[lhs_sampler(X, no_samples, seed)]
            T_train = np.atleast_2d(T_train).flatten().reshape(-1,1)
        
        # Uptake data
        Y = dataframe.iloc[:, uptake_index]
        Y_train = Y[lhs_sampler(X, no_samples, seed)]
        Y_train = np.atleast_2d(Y_train).flatten().reshape(-1,1)
        # print(Y_train.shape)

        # Uptake error 
        error = dataframe.iloc[:, error_index]
        error_train = error[lhs_sampler(X, no_samples, seed)]
        error_train = np.atleast_2d(error_train).flatten().reshape(-1,1)
        # print(error_train.shape)
        
        sample_counter = X_train.shape[0]
        if sample_counter < no_samples:
            no_samples += 1

    # Write the data to dataframes and write to csv
    if mol_frac_index is None and Temp_index is None:
        X_train = pd.DataFrame(X_train, columns =['Pressure [Pa]'])
        Y_train = pd.DataFrame(Y_train, columns = ['Uptake [mg/g]'])
        error_train = pd.DataFrame(error_train, columns = ['Error [mg/g]'])
        data_file = pd.concat([X_train, Y_train, error_train], axis = 1)
    elif mol_frac_index != None and Temp_index is None:
        X_train = pd.DataFrame(X_train, columns =['Pressure [Pa]'])
        X2_train = pd.DataFrame(X2_train, columns = ['Mole Frac'])
        Y_train = pd.DataFrame(Y_train, columns = ['Uptake [mg/g]'])
        error_train = pd.DataFrame(error_train, columns = ['Error [mg/g]'])
        data_file = pd.concat([X_train, X2_train, Y_train, error_train], axis = 1)
    elif mol_frac_index is None and Temp_index != None:
        X_train = pd.DataFrame(X_train, columns =['Pressure [Pa]'])
        T_train = pd.DataFrame(T_train, columns = ['Temp [K]'])
        Y_train = pd.DataFrame(Y_train, columns = ['Uptake [mg/g]'])
        error_train = pd.DataFrame(error_train, columns = ['Error [mg/g]'])
        data_file = pd.concat([X_train, T_train, Y_train, error_train], axis = 1)
    else:
        X_train = pd.DataFrame(X_train, columns =['Pressure [Pa]'])
        X2_train = pd.DataFrame(X2_train, columns = ['Mole Frac'])
        T_train = pd.DataFrame(T_train, columns = ['Temp [K]'])
        Y_train = pd.DataFrame(Y_train, columns = ['Uptake [mg/g]'])
        error_train = pd.DataFrame(error_train, columns = ['Error [mg/g]'])
        data_file = pd.concat([X_train, X2_train, T_train, Y_train, error_train], axis = 1)

    return data_file


class Transfer_GP:
    
    def __init__(self, config = {}, mean_function = None):
        kernel = config.get('kernel', 'RQ')
        usewhitekernel = config.get('useWhiteKernel', 'True')
        self.trainLikelihood = config.get('trainLikelihood', 'True')

        # select and initialize the kernels
        if kernel == 'RBF':
            gp_kernel = gpflow.kernels.SquaredExponential()
        if kernel == 'RQ':
            gp_kernel = gpflow.kernels.RationalQuadratic()
        if kernel == 'Matern12':
            gp_kernel = gpflow.kernels.Matern12()
        if kernel == 'Matern52':
            gp_kernel = gpflow.kernels.Matern52()

        # Add white kernel
        if usewhitekernel:
            gp_kernel = gp_kernel + gpflow.kernels.White()
        self.kernel = gp_kernel
        self.mean_function = mean_function
        self.model = None
    
    def fit(self, X, Y):

        # build gp model

        self.model = gpflow.models.GPR((X, Y), self.kernel, mean_function = self.mean_function, 
                                       noise_variance = 10 ** -2)
        
        gpflow.utilities.set_trainable(self.model.likelihood.variance,self.trainLikelihood)
        # build optimizer
        opt = gpflow.optimizers.Scipy()

        # fit GP to training data
        
        aux = opt.minimize(self.model.training_loss, self.model.trainable_variables, 
                           options = dict(maxiter = 2000, ftol = 1e-9), method = 'l-bfgs-b', 
                           step_callback = None, compile = True, allow_unused_variables = False)

        # check convergence
        if aux.success == False:
            warnings.warn('GP optimizer failed to converge')
        return self.model
    
    def predict(self, x_test):
        model = self.model
        # perform gp prediction and get mean and variance

        GP_mean, GP_var = model.predict_f(x_test)

        y_pred = GP_mean
        sigma = GP_var ** 0.5

        # output

        return y_pred, sigma