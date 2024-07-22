import numpy as np

"""
Standard numerical solver (bakward Euler scheme)
for wave equation
Input:
  ys - initial state
  dt - time step
  nsteps - number of steps to perform with solver
  all_steps - if true, return all steps. If false,
              only return last step.
Output:
  sol - The resulting solution from num solver  
"""
def num_step(ys, dt, nsteps, all_steps=True):
   dx = 1/(len(ys)-1)
   params = np.array([1 - dt / dx, dt/dx])
   A = make_matrix(params, len(ys), 2)
   sol = solver(A, ys, nsteps, stencil=2, all_steps=all_steps)
   return sol

"""
Numerical solver for wave equation using ML to learn params
Input:
  ys - initial state
  nsteps - number of steps to perform with solver
  params - the learned parameters to use in the matrix
           solver. The first entry goes on the diagonal,
           the second entry one to the left of the diag,
           and so on.
  stencil - the width of the stencil (e.g. how many
            upwind data points will be used in solver)
  all_steps - if true, return all steps. If false,
              only return last step.
Output:
  sol - The resulting solution from num solver  
"""
def ml_step(ys, nsteps, params, stencil, all_steps=True):
   nx = len(ys) + stencil - 2 # add extra ghost points if stencil>2
   ys_ghost = np.zeros(nx)
   ys_ghost[stencil-2:] = ys
   if stencil > 2:
      ys_ghost[:stencil-2] = ys[len(ys)-(stencil-2):len(ys)]
   A = make_matrix(params, nx, stencil)
   sol = solver(A, ys_ghost, nsteps, stencil, all_steps)
   return sol
    

"""
General solver given step matrix A to go from initial
state to next state
"""
def solver(A, ys, nsteps, stencil=2, all_steps=True):
   y_next = np.zeros((len(ys), nsteps+1))
   y_next[:,0] = ys

   for i in range(nsteps):
      y_next[:,i+1] = A @ y_next[:,i]
      y_next[:stencil-1,i+1] = y_next[-(stencil-1):,i+1]
   if all_steps:
      return y_next[stencil-2:,:]
   else:
      print(y_next.shape)
      return y_next[stencil-2:,-1]

"""
Construct the matrix A given the params.
nx is the number of points in the spatial domain.
If there is 1 less param than the stencil width,
choose the last param so as to satisfy that the sum
of all the params (a row in A) is 1.
"""
def make_matrix(params, nx, stencil):
   A = np.eye(nx)*params[0]
   for i in range(1,len(params)):
      A += np.diag(np.ones(nx-i)*params[i], k=-i)
   if len(params)+1 == stencil:
      A += np.diag(np.ones(nx-stencil)*(1-np.sum(params)), k=-stencil)
   return A