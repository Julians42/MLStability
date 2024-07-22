import numpy as np
from wave_solvers import ml_step
from scipy.optimize import minimize

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
def ml_loss(params, data, rollout, stencil, nx):    
   # mse for each rollout period (1,2,...,rollout)
   mse = np.zeros(rollout)
   n = np.zeros(rollout) # number of data points per rollout period
   
   # Iterate over each time step
   for ti in range(len(data[0])-1):        
      # Make predictions through the rollout period (if enough time points)
      nsteps = min(len(data[0])-1-ti, rollout)
      pred = ml_step(data[:,ti], nsteps, params, stencil)
      # Get the L2 error
      mse[:nsteps] += np.sum((pred[1:,1:] - data[1:,ti+1:ti+1+nsteps])**2, axis=0)
      n[:nsteps] += nx-1 # don't calculate loss from repeated point in loss
   
   # Get MSE
   mse /= n
   
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
def train_ml(dt, dx, nsteps, rollout, stencil, f, print_mes=True, params0=None):
   x = np.arange(0,1+dx/2, dx)
   t = np.arange(0,dt*nsteps+dt/2,dt)
   X, T = np.meshgrid(x, t, indexing='ij')
   data_train = f(X, T)
   nx = int(1/dx+1)
   
   if not params0:
      params0 = np.ones(stencil)/stencil
   result = minimize(ml_loss, params0, args=(data_train, rollout, stencil, nx), method='BFGS')
   params = result.x
   if print_mes:
      print('Success! Learned params:', params)
      print('Numerical a, b:', (1 - dt / dx), dt/dx) 
   return params
    