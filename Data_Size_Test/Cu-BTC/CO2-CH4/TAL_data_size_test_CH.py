import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os
from sklearn.metrics import r2_score, mean_absolute_error
import GPR
import gpflow
import tensorflow as tf

#========= Training Data Generation ==========
# read in the complete GCMC ground truth data
df = pd.read_csv('Loadings_CH4_Cu-BTC-2.csv')
df2 = pd.read_excel('complete_Krish_1.xlsx')
df3 = pd.read_csv('complete_Krish.csv')
# print(df.head())
# print(df2.head())
# print(df3.head())

# Generate training data for both pure and mixture and write to csv
data_pure = GPR.train_data_generator(df, 15, 40, 0, 1, 2)
print(data_pure.shape)
data_pure.to_csv('training_pure_ch.csv', index= False)

#======= Commence Pure Data Model development =======

# Read in the pure CH4 training data
gf = pd.read_csv('training_pure_ch.csv')

# define training data and label
x = gf.iloc[:, 0]
y = gf.iloc[:, 1]

x_test = df.iloc[:, 0]
x_g = df.iloc[:, 0]

# normalize data
x = np.atleast_2d(x).flatten().reshape(-1,1)
y = np.atleast_2d(y).flatten().reshape(-1,1)
x_test = np.atleast_2d(x_test).flatten().reshape(-1,1)

x = x /1e5
x_test = x_test/1e5
x_g = x_g/1e5

x_s = GPR.normalize(x, skScaler=x_g, method = 'LogStandardization')
x_test = GPR.normalize(x_test, x_g, 'LogStandardization')
y = GPR.normalize(y, x_g, 'Log')

# pad training and test data to desired dimensions
dim = 3

x_st = GPR.data_padding(x_s, dim)
x_test = GPR.data_padding(x_test, dim)

# print(x_st)
# print(x_test)

# Define gp configurations
gpConfig={'kernel':'RBF',
          'useWhiteKernel':True,
          'trainLikelihood':True}

max_iter = 500
lim = 0.02
factor = 1e-3

# AL loop

for i in range(max_iter):

    model = GPR.gp(x_st, y, config = gpConfig)

    y_pred_CH4, sigma_CH4 = GPR.gp_predict(model, x_test)

    rel_error_CH4 = np.zeros(len(sigma_CH4))

    for i in range(len(sigma_CH4)):
        rel_error_CH4[i] = abs(sigma_CH4[i])/abs(y_pred_CH4[i])
    print(rel_error_CH4)
    
    max_CH4 = np.max(rel_error_CH4)
    index_CH4 = np.argmax(rel_error_CH4)
    
    if max_CH4 < lim:
        data = x_test[index_CH4]
        print('Done')
        print(f'Final maximum error for CH4 is {rel_error_CH4[index_CH4]}')
        ch4_model = model
        break
    else:
        data = x_test[index_CH4]

        x_st = np.append(x_st, data)
        x_st = x_st.reshape(-1,dim)

        x_test = np.delete(x_test, index_CH4, axis= 0)
        x_test = np.atleast_2d(x_test).flatten().reshape(-1, dim)

        data = data.reshape(-1, dim)
        data_s = GPR.reverse(data[0,0], skScaler = x_g, method = 'LogStandardization')
        data_s = 1e5 * data_s
        data_s = data_s.flatten()
        data_s = data_s[0]
        data_s = round(data_s)
        y_data = df[round(df['Pressure [Pa]']) == data_s]['Uptake [mg/g]']
        y_data = np.atleast_2d(y_data).flatten().reshape(-1,1)
        y_data = GPR.normalize(y_data, skScaler = x_g, method = 'Log')
        
        y = np.append(y, y_data)
        y = y.reshape(-1,1)

# Define mean function from source model
def mean_gen(x, model = ch4_model):
    
    y_mean, __ = GPR.gp_predict(model, x)
    
    return y_mean

class SourceMean(gpflow.mean_functions.MeanFunction):
        
    def __call__(self, X):
        X = tf.convert_to_tensor(X)
        return mean_gen(X)
    
mean_calc = SourceMean()

#======= Commence training for CH4 in P-X space with different training data size =======
data_size_px = [5, 10, 15, 20, 25, 30, 35, 40]

# Define gp configurations
gpConfig={'kernel':'RBF',
          'useWhiteKernel':True,
          'trainLikelihood':True}

max_iter = 2000
lim = 0.02
factor = 1e-3

err_coll_CH4 = {}

for i in data_size_px:
    data_mix = GPR.train_data_generator(df2, i, 40, 0, 5, 7, 3)
    print(data_mix.shape)

    # Get actual number of data generated
    actual_len = data_mix.shape[0]

    # Define training and test data and standardize
    p = data_mix.loc[:, 'Pressure [Pa]']
    x2 = data_mix.loc[:, 'Mole Frac']
    y2 = data_mix.loc[:, 'Uptake [mg/g]']

    p_test = df2.loc[:, 'Pressure(Pa)']
    p_g = df2.loc[:, 'Pressure(Pa)']
    x2_test = df2.loc[:, 'CH4_mole_frac']
    x2_test = x2_test.dropna()
    x2_g = df2.loc[:, 'CH4_mole_frac']
    x2_g = x2_g.dropna()

    # convert 1D to 2D array
    p = np.atleast_2d(p).flatten().reshape(-1,1)
    x2 = np.atleast_2d(x2).flatten().reshape(-1,1)
    y2 = np.atleast_2d(y2).flatten().reshape(-1,1)

    p_test = np.atleast_2d(p_test).flatten().reshape(-1,1)
    x2_test = np.atleast_2d(x2_test).flatten().reshape(-1,1)

    # Replacing y if some y values are zeroes

    for i in range(len(y2)):
        if y2[i] == 0:
            y2[i] = 0.0001

    # Converting Pressure to bars

    p = p/(1.0e5)
    p_test = p_test/(1.0e5)
    p_g = p_g/(1.0e5)

    # normalize data

    p_s = GPR.normalize(p, p_g, 'LogStandardization')
    p_test = GPR.normalize(p_test, p_g, 'LogStandardization')

    x2 = GPR.normalize(x2, x2_g, 'MinMaxNorm')
    x2_test = GPR.normalize(x2_test, x2_g, 'MinMaxNorm')

    y2 = GPR.normalize(y2, p_g, 'Log')

    # Define scaled training and test data for training GP
    x_s2 = np.zeros((len(p_s), dim))

    x_test_CH4 = np.zeros((len(p_test), dim))

    # Fill the training and test data with both pressure and mole fraction

    for i in range(len(p_s)):
        for j in range(2):
            # Insert pressure in the first column and linearly standardized mole fraction in the second
            if j == 0:
                x_s2[i, j] = p_s[i]
            else:
                x_s2[i, j] = x2[i]

    for i in range(len(p_test)):
        for k in range(2):
            if k == 0:
                x_test_CH4[i, k] = p_test[i]
            else:
                x_test_CH4[i, k] = x2_test[i]

    err_coll = []

    # AL loop

    for j in range(max_iter):

        model = GPR.gp(x_s2, y2, config = gpConfig, mean_func= mean_calc)

        y_pred_CH4_M, sigma_CH4_M = GPR.gp_predict(model, x_test_CH4)

        rel_error_CH4_M = np.zeros(len(sigma_CH4_M))

        for i in range(len(sigma_CH4_M)):
            rel_error_CH4_M[i] = abs(sigma_CH4_M[i])/abs(y_pred_CH4_M[i])
        print(rel_error_CH4_M)

        err_coll.append(np.mean(rel_error_CH4_M))
        
        max_CH4_M = np.max(rel_error_CH4_M)
        index_CH4_M = np.argmax(rel_error_CH4_M)
        
        if max_CH4_M < lim:
            print('Done')
            print(f'Final maximum error for CH4 in mixture is {rel_error_CH4_M[index_CH4_M]}')
            ch4_model_px = model
            break
        else:
            data = x_test_CH4[index_CH4_M]

            x_s2 = np.append(x_s2, data)
            x_s2 = x_s2.reshape(-1,dim)

            x_test_CH4 = np.delete(x_test_CH4, index_CH4_M, axis= 0)
            x_test_CH4 = np.atleast_2d(x_test_CH4).flatten().reshape(-1, dim)

            data = data.reshape(-1, dim)
            data_p = GPR.reverse(data[0,0], skScaler = p_g, method = 'LogStandardization')
            data_p = 1e5 * data_p
            data_p = data_p.flatten()
            data_p = data_p[0]
            data_p = round(data_p)
            data_x = GPR.reverse(data[0, 1], x2_g, 'MinMaxNorm')
            data_x = round(data_x, 2)
            y_data = df2[(round(df2['Pressure(Pa)']) == data_p) & (round(df2['CH4_mole_frac'], 2) == data_x)]['CH4_uptake']
            y_data = np.atleast_2d(y_data).flatten().reshape(-1,1)

            if (y_data <= 0).any():
                y_data = 0.0001
                y_data = np.atleast_2d(y_data).flatten().reshape(-1,1)
                y_data = GPR.normalize(y_data, p_g, 'Log')
            else:
                y_data = y_data + 0
                y_data = np.atleast_2d(y_data).flatten().reshape(-1,1)
                y_data = GPR.normalize(y_data, p_g, 'Log')
            
            y2 = np.append(y2, y_data)
            y2 = y2.reshape(-1,1)
    
    # Define data for making predictions on our model
    p_pred = np.atleast_2d(p_g).flatten().reshape(-1,1)
    x2_pred = np.atleast_2d(x2_g).flatten().reshape(-1,1)

    p_pred = GPR.normalize(p_pred, skScaler = p_g, method = 'LogStandardization')
    x2_pred = GPR.normalize(x2_pred, x2_g, 'MinMaxNorm')

    x_test_pred = np.zeros((len(p_pred), dim))

    for i in range(len(p_pred)):
        for k in range(2):
            if k == 0:
                x_test_pred[i, k] = p_pred[i]
            else:
                x_test_pred[i, k] = x2_pred[i]

    # make predictions on the data using the trained model
    GP_pred, GP_var = GPR.gp_predict(ch4_model_px, x_test_pred)

    GP_pred = GPR.reverse(GP_pred, skScaler = p_g, method = 'Log')
    GP_pred = GP_pred.numpy()

    # Calculate metrics for the model and write to file
    y_g = df2.loc[:, 'CH4_uptake']
    err_coll_CH4[i] = err_coll
    
    for i in range(len(y_g)):
        if y_g[i] == 0:
            y_g[i] = 0.0001

    r_squared = r2_score(y_g,GP_pred)
    print(r_squared)
    MAE = mean_absolute_error(y_g,GP_pred)
    print(MAE)

    act_RE = np.zeros(len(GP_pred))

    for i in range(len(GP_pred)):
        act_RE[i] = abs((GP_pred[i] - y_g[i])/y_g[i])

        MRE = np.mean(act_RE)
    print(MRE)

    no_iters = len(err_coll_CH4[i])
    print(no_iters)

    os.system("echo "+str(actual_len)+", "+str(r_squared)+", "+str(MAE)+", "+str(MRE)+", "+str(no_iters)+"  >> Metrics_pure_ch_ch_px.csv")


#======= Commence Training for CH4 in P-X-T Space ========
data_size_pxt = [5, 10, 15, 20, 25, 30, 40, 50]

# Define gp configurations
gpConfig={'kernel':'RBF',
          'useWhiteKernel':True,
          'trainLikelihood':True}

max_iter = 2000
lim = 0.02
factor = 1e-3
err_coll_CH4_pxt = {}

for i in data_size_pxt:
    data_mix_pxt = GPR.train_data_generator(df3, i, 40, 0, 5, 7, 3, 1)
    print(data_mix_pxt.shape)

    # Get actual number of data generated
    actual_len_pxt = data_mix_pxt.shape[0]
    # Define training and test data and standardize

    p_pxt = data_mix_pxt.loc[:, 'Pressure [Pa]']
    t = data_mix_pxt.loc[:, 'Temp [K]']
    x2_pxt = data_mix_pxt.loc[:, 'Mole Frac']
    y2_pxt = data_mix_pxt.loc[:, 'Uptake [mg/g]']

    p_test_pxt = df3.loc[:, 'Pressure(Pa)']
    p_g_pxt = df3.loc[:, 'Pressure(Pa)']
    x2_test_pxt = df3.loc[:, 'CH4_mole_frac']
    x2_test_pxt = x2_test_pxt.dropna()
    x2_g_pxt = df3.loc[:, 'CH4_mole_frac']
    x2_g_pxt = x2_g_pxt.dropna()
    t_test = df3.loc[:, 'Temperature(K)']
    t_g = df3.loc[:, 'Temperature(K)']

    # convert 1D to 2D array
    p_pxt = np.atleast_2d(p_pxt).flatten().reshape(-1,1)
    t = np.atleast_2d(t).flatten().reshape(-1,1)
    x2_pxt = np.atleast_2d(x2_pxt).flatten().reshape(-1,1)
    y2_pxt = np.atleast_2d(y2_pxt).flatten().reshape(-1,1)

    p_test_pxt = np.atleast_2d(p_test_pxt).flatten().reshape(-1,1)
    x2_test_pxt = np.atleast_2d(x2_test_pxt).flatten().reshape(-1,1)
    t_test = np.atleast_2d(t_test).flatten().reshape(-1,1)

    # Replacing y if some y values are zeroes

    for i in range(len(y2_pxt)):
        if y2_pxt[i] == 0:
            y2_pxt[i] = 0.0001

    # Converting Pressure to bars

    p_pxt = p_pxt/(1.0e5)
    p_test_pxt = p_test_pxt/(1.0e5)
    p_g_pxt = p_g_pxt/(1.0e5)

    # normalize data

    p_s_pxt = GPR.normalize(p_pxt, p_g_pxt, 'LogStandardization')
    p_test_pxt = GPR.normalize(p_test_pxt, p_g_pxt, 'LogStandardization')

    t_s = GPR.normalize(t, t_g, 'LogStandardization')
    t_test = GPR.normalize(t_test, t_g, 'LogStandardization')

    x2_pxt = GPR.normalize(x2_pxt, x2_g_pxt, 'MinMaxNorm')
    x2_test_pxt = GPR.normalize(x2_test_pxt, x2_g_pxt, 'MinMaxNorm')

    y2_pxt = GPR.normalize(y2_pxt, p_g_pxt, 'Log')

    # Define scaled training and test data for training GP
    x_s2_pxt = np.zeros((len(p_s_pxt), dim))

    x_test_CH4_pxt = np.zeros((len(p_test_pxt), dim))

    # Fill the training and test data with both pressure and mole fraction

    for i in range(len(p_s_pxt)):
        for j in range(dim):
            # Insert pressure in the first column and linearly standardized mole fraction in the second
            if j == 0:
                x_s2_pxt[i, j] = p_s_pxt[i]
            elif j == 1:
                x_s2_pxt[i, j] = t_s[i]
            else:
                x_s2_pxt[i, j] = x2_pxt[i]

    for i in range(len(p_test_pxt)):
        for k in range(dim):
            if k == 0:
                x_test_CH4_pxt[i, k] = p_test_pxt[i]
            elif k == 1:
                x_test_CH4_pxt[i, k] = t_test[i]
            else:
                x_test_CH4_pxt[i, k] = x2_test_pxt[i]

    # AL loop
    
    err_coll_pxt = []

    for j in range(max_iter):

        model = GPR.gp(x_s2_pxt, y2_pxt, config = gpConfig, mean_func= mean_calc)

        y_pred_CH4_pxt, sigma_CH4_pxt = GPR.gp_predict(model, x_test_CH4_pxt)

        rel_error_CH4_pxt = np.zeros(len(sigma_CH4_pxt))

        for i in range(len(sigma_CH4_pxt)):
            rel_error_CH4_pxt[i] = abs(sigma_CH4_pxt[i])/abs(y_pred_CH4_pxt[i])
        print(rel_error_CH4_pxt)

        err_coll_pxt.append(np.mean(rel_error_CH4_pxt))
        
        max_CH4_pxt = np.max(rel_error_CH4_pxt)
        index_CH4_pxt = np.argmax(rel_error_CH4_pxt)

        if max_CH4_pxt < lim:
            print('Done')
            print(f'Final maximum error for CH4 in mixture is {rel_error_CH4_pxt[index_CH4_pxt]}')
            ch4_model_pxt = model
            break
        else:
            data = x_test_CH4_pxt[index_CH4_pxt]

            x_s2_pxt = np.append(x_s2_pxt, data)
            x_s2_pxt = x_s2_pxt.reshape(-1,dim)

            x_test_CH4_pxt = np.delete(x_test_CH4_pxt, index_CH4_pxt, axis= 0)
            x_test_CH4_pxt = np.atleast_2d(x_test_CH4_pxt).flatten().reshape(-1, dim)

            data = data.reshape(-1, dim)
            data_p = GPR.reverse(data[0,0], skScaler = p_g_pxt, method = 'LogStandardization')
            data_p = 1e5 * data_p
            data_p = data_p.flatten()
            data_p = data_p[0]
            data_p = round(data_p)
            data_t = GPR.reverse(data[0, 1], t_g, 'LogStandardization')
            data_t = round(data_t)
            data_x = GPR.reverse(data[0, 2], x2_g_pxt, 'MinMaxNorm')
            data_x = round(data_x, 2)
            y_data = df3[(round(df3['Pressure(Pa)']) == data_p) & (df3['Temperature(K)'] == data_t) & (round(df3['CH4_mole_frac'], 2) == data_x)]['CH4_uptake']
            y_data = np.atleast_2d(y_data).flatten().reshape(-1,1)

            if (y_data <= 0).any():
                y_data = 0.0001
                y_data = np.atleast_2d(y_data).flatten().reshape(-1,1)
                y_data = GPR.normalize(y_data, p_g_pxt, 'Log')
            else:
                y_data = y_data + 0
                y_data = np.atleast_2d(y_data).flatten().reshape(-1,1)
                y_data = GPR.normalize(y_data, p_g_pxt, 'Log')
            
            y2_pxt = np.append(y2_pxt, y_data)
            y2_pxt = y2_pxt.reshape(-1,1)

    # Define data for making predictions on our model
    p_pred_pxt = np.atleast_2d(p_g_pxt).flatten().reshape(-1,1)
    t_pred = np.atleast_2d(t_g).flatten().reshape(-1,1)
    x2_pred_pxt = np.atleast_2d(x2_g_pxt).flatten().reshape(-1,1)

    p_pred_pxt = GPR.normalize(p_pred_pxt, skScaler = p_g_pxt, method = 'LogStandardization')
    t_pred = GPR.normalize(t_pred, t_g, 'LogStandardization')
    x2_pred_pxt = GPR.normalize(x2_pred_pxt, x2_g_pxt, 'MinMaxNorm')

    x_test_pred_pxt = np.zeros((len(p_pred_pxt), dim))

    for i in range(len(p_pred_pxt)):
        for k in range(dim):
            if k == 0:
                x_test_pred_pxt[i, k] = p_pred_pxt[i]
            elif k == 1:
                x_test_pred_pxt[i, k] = t_pred[i] 
            else:
                x_test_pred_pxt[i, k] = x2_pred_pxt[i]

    # make predictions on the data using the trained model
    GP_pred_pxt, GP_var_pxt = GPR.gp_predict(ch4_model_pxt, x_test_pred_pxt)

    GP_pred_pxt = GPR.reverse(GP_pred_pxt, skScaler = p_g_pxt, method = 'Log')
    GP_pred_pxt = GP_pred_pxt.numpy()

    # Calculate metrics for the model and write to file
    y_g_pxt = df3.loc[:, 'CH4_uptake']

    err_coll_CH4_pxt[i] = err_coll_pxt

    for i in range(len(y_g_pxt)):
        if y_g_pxt[i] == 0:
            y_g_pxt[i] = 0.0001

    r_squared_pxt = r2_score(y_g_pxt,GP_pred_pxt)
    print(r_squared_pxt)
    MAE_pxt = mean_absolute_error(y_g_pxt,GP_pred_pxt)
    print(MAE_pxt)

    act_RE_pxt = np.zeros(len(GP_pred_pxt))

    for i in range(len(GP_pred_pxt)):
        act_RE_pxt[i] = abs((GP_pred_pxt[i] - y_g_pxt[i])/y_g_pxt[i])

        MRE_pxt = np.mean(act_RE_pxt)
    print(MRE_pxt)

    no_iters_pxt = len(err_coll_CH4_pxt[i])
    print(no_iters_pxt)

    os.system("echo "+str(actual_len_pxt)+", "+str(r_squared_pxt)+", "+str(MAE_pxt)+", "+str(MRE_pxt)+", "+str(no_iters_pxt)+"  >> Metrics_pure_ch_ch_pxt.csv")



