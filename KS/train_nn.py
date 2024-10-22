import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from KS_solver import KS_step

class Model(nn.Module):
    def __init__(self, stencil, hidden_dim):
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.stencil = stencil
        
        self.fc1 = nn.Linear(stencil, 100).double()
        self.fc2 = nn.Linear(100, 50).double()
        self.fc3 = nn.Linear(50, 1).double()
        self.act = nn.ReLU()
    
    def forward(self, u):
        uout = self.fc3(self.act(self.fc2(self.act(self.fc1(u)))))
        # uout = torch.sum(u*params, dim=1)
        
        return uout


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
def train_nn(KS, dt, nx, rollout, stencil, f_rand, print_mes=True, nb=5):
    L = KS.l
    x = np.arange(0,L, L/nx)

    # Instantiate the model with hyperparameters
    model = Model(stencil=stencil, hidden_dim=100)


    t_weights = torch.tensor(np.arange(1,rollout+1)).unsqueeze(0)

    # Define hyperparameters
    n_epochs = 10000
    lr = 0.001

    # Define Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    T = rollout+stencil

    with torch.autograd.set_detect_anomaly(True):
    
        for epoch in range(0, n_epochs):
            optimizer.zero_grad()

            target_seq = np.zeros((nb, rollout+stencil, len(x)))
            for i in range(nb):
                f = f_rand(KS.l)
                target_seq[i,:,:] = KS_step(KS, rollout+stencil-1, f(x))    
            target_seq = target_seq.reshape(-1, target_seq.shape[1])
            
            target_seq = torch.tensor(target_seq, dtype=torch.double)
            output = torch.zeros(target_seq.shape, dtype=torch.double)

            output[:,:stencil] = target_seq[:,:stencil]
            for i in range(stencil,rollout+stencil):
                f_t = model(output[:,i-stencil:i].clone())
                output[:,i] = f_t.squeeze()

            
            loss = torch.mean((output[:,stencil:] - target_seq[:,stencil:])**2/t_weights) # don't include IC
            print(epoch, 'loss: ', np.round(loss.detach().numpy(),6))
            loss.backward() # Does backpropagation and calculates gradients
            optimizer.step() # Updates the weights accordingly
            
    return model

# inputted as dimensions (time, x)
def nn_step(model, y0, num_steps, stencil, all_steps=True):
    if len(y0.shape)==1:
        y0 = np.expand_dims(y0, 0)
    else:
        y0 = y0.T
    assert(stencil == y0.shape[1])
    y_next = np.zeros((y0.shape[0], stencil+num_steps))
    y_next[:,:stencil] = y0
    y_next = torch.tensor(y_next)
    for i in range(stencil,num_steps+stencil):
        params, f_t = model(y_next[:,i-stencil:i])
        y_next[:,i] = f_t

    # transpose back to (time,x)
    y_next = y_next.detach().numpy().T
    
    if all_steps:
        return y_next
    else:
        return y_next[-1]