import numpy as np
from scipy.optimize import minimize
from KS_solver import KS_step, ml_step, make_matrix

"""
Loss function to train the ML model
The L2 error of the model against the ground truth averaged across
the different rollout periods
Inputs:
   params - the parameters for the solver matrix which will be optimized
   data - the ground truth data at all evenly spaced time points
   rollout - the number of time steps past an input we will consider up to
   stencil - the stencil width for the solver matrix
   nx - the number of points in our spatial domain
Outputs:
   mse - the mse averaged over our different rollout periods
"""
def ml_loss(params, data, rollout, stencil_x, stencil_t): 
   # mse for each rollout period (1,2,...,rollout)
   mse = np.zeros(rollout)
   
   # Iterate over each time step
   # Make predictions through the rollout period (if enough time points)
   for b in range(len(data)):
      pred = ml_step(data[b,:stencil_t,:], rollout, params, stencil_x)
      # Get the L2 error
      mse += np.sum((pred[stencil_t:] - data[b,stencil_t:,:])**2, axis=1)
   
   # print('MSE', np.round(mse[0], 4), np.round(mse[50], 4))
      
   # Get MSE
   mse /= np.arange(1,rollout+1)

   # print('MSE', np.round(mse[1], 4), np.round(mse[50], 4))
   
   # Equally weight the losses across the different rollout periods
   mse = np.mean(mse)

   return mse


"""
Loss function to train the ML model
The L2 error of the model against the ground truth averaged across
the different rollout periods
Inputs:
   params - the parameters for the solver matrix which will be optimized
   data - the ground truth data at all evenly spaced time points
   rollout - the number of time steps past an input we will consider up to
   stencil - the stencil width for the solver matrix
   nx - the number of points in our spatial domain
Outputs:
   mse - the mse averaged over our different rollout periods
"""        
def train_ml(KS, dt, nx, rollout, stencil_x, stencil_t, f_rand, print_mes=True, nb=5):
    L = KS.l
    x = np.arange(0,L, L/nx)
    data_train = np.zeros((nb, rollout+stencil_t, len(x)))
    for i in range(nb):
        f = f_rand(KS.l)
        data_train[i,:,:] = KS_step(KS, rollout+stencil_t-1, f(x)) 
    

    params0 = np.ones(stencil_x + stencil_t - 1)/(stencil_x+stencil_t - 1)

    result = minimize(ml_loss, params0, args=(data_train, rollout, stencil_x, stencil_t), method='CG')
    params = result.x
    if print_mes:
        print('Success! Learned params:', params)
    return params
    
    