import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os
from sklearn.metrics import r2_score, mean_absolute_error
import GPR
import gpflow
import tensorflow as tf

# Read in the mixture training and complete data
df2 = pd.read_csv('complete_Krish.csv')
print(df2.head())

hf = pd.read_csv('training_mix_pxt_ch.csv')
print(hf.head())

# Define training and test data and standardize

p = hf.loc[:, 'Pressure [Pa]']
t = hf.loc[:, 'Temp [K]']
x2 = hf.loc[:, 'Mole Frac']
y2 = hf.loc[:, 'Uptake [mg/g]']

p_test = df2.loc[:, 'Pressure(Pa)']
p_g = df2.loc[:, 'Pressure(Pa)']
x2_test = df2.loc[:, 'CH4_mole_frac']
x2_test = x2_test.dropna()
x2_g = df2.loc[:, 'CH4_mole_frac']
x2_g = x2_g.dropna()
t_test = df2.loc[:, 'Temperature(K)']
t_g = df2.loc[:, 'Temperature(K)']

# convert 1D to 2D array
p = np.atleast_2d(p).flatten().reshape(-1,1)
t = np.atleast_2d(t).flatten().reshape(-1,1)
x2 = np.atleast_2d(x2).flatten().reshape(-1,1)
y2 = np.atleast_2d(y2).flatten().reshape(-1,1)

p_test = np.atleast_2d(p_test).flatten().reshape(-1,1)
x2_test = np.atleast_2d(x2_test).flatten().reshape(-1,1)
t_test = np.atleast_2d(t_test).flatten().reshape(-1,1)

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

t_s = GPR.normalize(t, t_g, 'LogStandardization')
t_test = GPR.normalize(t_test, t_g, 'LogStandardization')

x2 = GPR.normalize(x2, x2_g, 'MinMaxNorm')
x2_test = GPR.normalize(x2_test, x2_g, 'MinMaxNorm')

y2 = GPR.normalize(y2, p_g, 'Log')

# Define scaled training and test data for training GP
dim = 3
x_s2 = np.zeros((len(p_s), dim))

x_test_CH4 = np.zeros((len(p_test), dim))

# Fill the training and test data with both pressure and mole fraction

for i in range(len(p_s)):
    for j in range(dim):
        # Insert pressure in the first column and linearly standardized mole fraction in the second
        if j == 0:
            x_s2[i, j] = p_s[i]
        elif j == 1:
            x_s2[i, j] = t_s[i]
        else:
            x_s2[i, j] = x2[i]

for i in range(len(p_test)):
    for k in range(dim):
        if k == 0:
            x_test_CH4[i, k] = p_test[i]
        elif k == 1:
            x_test_CH4[i, k] = t_test[i]
        else:
            x_test_CH4[i, k] = x2_test[i]

# Define gp configurations
gpConfig={'kernel':'RBF',
          'useWhiteKernel':True,
          'trainLikelihood':True}

max_iter = 2000
lim = 90
factor = 1e-3
err_coll_CH4_M = []
max_coll_CH4_M = []
pac_coll_CH4_pxt = []

# AL loop

for i in range(max_iter):

    conf_CH4_pxt = 0
    nconf_CH4_pxt = 0

    model = GPR.gp(x_s2, y2, config = gpConfig)

    y_pred_CH4_M, sigma_CH4_M = GPR.gp_predict(model, x_test_CH4)

    rel_error_CH4_M = np.zeros(len(sigma_CH4_M))

    for i in range(len(sigma_CH4_M)):
        rel_error_CH4_M[i] = abs(sigma_CH4_M[i])/abs(y_pred_CH4_M[i])

        if rel_error_CH4_M[i] < 0.02:
            conf_CH4_pxt += 1
        else:
            nconf_CH4_pxt += 1
    print(rel_error_CH4_M)

    err_coll_CH4_M.append(np.mean(rel_error_CH4_M))
    
    max_CH4_M = np.max(rel_error_CH4_M)
    index_CH4_M = np.argmax(rel_error_CH4_M)

    max_coll_CH4_M.append(max_CH4_M)

    PAC_CH4_pxt = 100*(conf_CH4_pxt/(conf_CH4_pxt + nconf_CH4_pxt))
    # print(PAC_CH4)
    pac_coll_CH4_pxt.append(PAC_CH4_pxt)
    
    if PAC_CH4_pxt >= lim:
        print('Done')
        print(f'Final maximum error for CH4 in mixture is {rel_error_CH4_M[index_CH4_M]}')
        ch4_model_m = model
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
        data_t = GPR.reverse(data[0, 1], t_g, 'LogStandardization')
        data_t = round(data_t)
        data_x = GPR.reverse(data[0, 2], x2_g, 'MinMaxNorm')
        data_x = round(data_x, 2)
        y_data = df2[(round(df2['Pressure(Pa)']) == data_p) & (df2['Temperature(K)'] == data_t) & (round(df2['CH4_mole_frac'], 2) == data_x)]['CH4_uptake']
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
t_pred = np.atleast_2d(t_g).flatten().reshape(-1,1)
x2_pred = np.atleast_2d(x2_g).flatten().reshape(-1,1)

p_pred = GPR.normalize(p_pred, skScaler = p_g, method = 'LogStandardization')
t_pred = GPR.normalize(t_pred, t_g, 'LogStandardization')
x2_pred = GPR.normalize(x2_pred, x2_g, 'MinMaxNorm')

x_test_pred = np.zeros((len(p_pred), dim))

for i in range(len(p_pred)):
    for k in range(dim):
        if k == 0:
            x_test_pred[i, k] = p_pred[i]
        elif k == 1:
            x_test_pred[i, k] = t_pred[i] 
        else:
            x_test_pred[i, k] = x2_pred[i]

# make predictions on the data using the trained model
GP_pred, GP_var = GPR.gp_predict(ch4_model_m, x_test_pred)

GP_pred = GPR.reverse(GP_pred, skScaler = p_g, method = 'Log')
GP_pred = GP_pred.numpy()

# write predictions to file
x_test_pred[:, 0]  = GPR.reverse(x_test_pred[:, 0], p_g, "LogStandardization")
x_test_pred[:, 1] = GPR.reverse(x_test_pred[:, 1], t_g, "LogStandardization")
x_test_pred[:, 2] = GPR.reverse(x_test_pred[:, 2], x2_g, 'MinMaxNorm')

a = pd.DataFrame(GP_pred, columns=['Predicted'])
b = pd.DataFrame(x_test_pred, columns=['X Test', 'Temperature', 'mole fraction'])
c = pd.concat([b, a], axis = 1)
c.to_csv('Pred_AL_CH_pxt.csv',index=False)

# write the updated training data to file
x_s2[:,0] = GPR.reverse(x_s2[:,0], p_g, 'LogStandardization')
x_s2[:, 0] = x_s2[:, 0] * 1e5
x_s2[:, 1] = GPR.reverse(x_s2[:,1], t_g, 'LogStandardization')
x_s2[:, 2] = GPR.reverse(x_s2[:,2], x2_g, 'MinMaxNorm')

d = pd.DataFrame(x_s2, columns= ['Pressure', 'Temperature', 'Mole Frac'])

y2 = GPR.reverse(y2, p_g, 'Log')
e = pd.DataFrame(y2, columns = ['Uptake'])
f = pd.concat([d, e], axis = 1)
f.to_csv('AL_ch_sampled points_pxt.csv', index = False)

# write the error collected to a file
g = pd.DataFrame(err_coll_CH4_M, columns = ['GP MRE'])
g.to_csv('AL_CH_MRE_pxt.csv', index = False)

# write the PAC collected to a file
p = pd.DataFrame(pac_coll_CH4_pxt, columns = ['PAC'])
p.to_csv('AL_CH_PAC_pxt.csv', index = False)

# Calculate metrics for the model and write to file
y_g = df2.loc[:, 'CH4_uptake']

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

no_iters = len(err_coll_CH4_M)
print(no_iters)

os.system("echo "+str(r_squared)+", "+str(MAE)+", "+str(MRE)+", "+str(no_iters)+"  >> Metrics_AL_ch_pxt.csv")