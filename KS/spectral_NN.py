
# Authors: Ryan Eusebi and Julian Schmitt
# Fully connected neural network operating in Fourier space for time stepping equations
# Select the number of modes you wish to model the equations

class Model(nn.Module):
    def __init__(self, hidden_dim, nmodes_full, nmodes, mode_mean):
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.nmodes_full = nmodes_full
        self.nmodes = nmodes
        self.mode_mean = mode_mean

        
        self.fc1 = nn.Linear(nmodes*2, hidden_dim).double()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim).double()
        self.fc3 = nn.Linear(hidden_dim,nmodes*2).double()
        self.act = nn.ELU()
    
    def forward(self, u):
        uout = self.fc3(self.act(self.fc2(self.act(self.fc1(u)))))
        
        return uout
    
    def prep_data(self, x, nmodes):
        x = get_fft(x, nmodes)
        x = x/self.mode_mean
        return x

    def inv_prep_data(self, y, nmodes_out):
        nx = y.shape[-1]//2+1
        y_shape = np.array(y.shape)
        y_shape[-1] = nmodes_out+1
        y_real = np.zeros(y_shape)

        y = y*self.mode_mean
        y_real[...,:nx] = y[...,:nx]
        y_im = y_real*0
        y_im[...,1:nx-1] = y[...,nx:]
        y = np.real(np.fft.irfft(y_real + y_im*1j))
        return y
    
def get_fft(x, nmodes):
    x = np.fft.rfft(x)
    x_real = np.real(x)
    x_im = np.imag(x)
    x = np.concatenate((x_real[...,:nmodes+1], x_im[...,1:nmodes]), axis=len(x.shape)-1)
    return x

# to use full spectrum of frequencies for an even nx, nmodes should be nmodes=nx/2 (the 0 mode is automatically included)
def train_nn(KS, dt, nx, nmodes, rollout,train_period, stencil, f_rand, n_epochs=10000,print_mes=True, nb=5):
    L = KS.l
    x = np.arange(0,L, L/nx)
    nmodes_full = nx//2
    assert(nx%2==0) #only positive nx

    # Get statistics of modes for processing data
    example = np.zeros((20, train_period+1, len(x)))
    for i in range(20):
        f = f_rand(KS.l)
        example[i,:,:] = KS_step(KS, train_period, f(x)) 
    example = get_fft(example, nmodes)
    mode_mean = np.mean(np.abs(example), axis=(0,1))
    weights = mode_mean**(1/12)
    mode_weights = torch.tensor(weights/np.sum(weights))
    print(mode_weights)

    # Instantiate the model with hyperparameters
    model = Model(100, nmodes_full, nmodes, mode_mean)

    # Define hyperparameters
    lr = 0.001

    # Define Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    with torch.autograd.set_detect_anomaly(True):
    
        for epoch in range(0, n_epochs):
            optimizer.zero_grad()

            target_seq = np.zeros((nb, train_period+1, len(x)))
            for i in range(nb):
                f = f_rand(KS.l)
                target_seq[i,:,:] = KS_step(KS, train_period, f(x)) 

             # imaginary coefficients for first and last must be real, so don't model them
            target_seq = model.prep_data(target_seq, nmodes)
            X = torch.tensor(target_seq[:,:-1].reshape(-1, target_seq.shape[2]))
            Y = torch.tensor(target_seq[:,1:].reshape(-1, target_seq.shape[2]))
            output = model(X)

            X_rollout = torch.tensor(X[:-10:2,:])
            Y_rollout = torch.tensor(X[5:-5:2,:])
            Y_rollout2 = torch.tensor(X[15:-5:2,:])

            Yi = X_rollout
            for i in range(rollout):
                Yi = model(Yi).clone()

            Yj = Yi[:-5].clone()
            for j in range(10):
                Yj = model(Yj).clone()
            
            loss = torch.mean((Y-output)**2*mode_weights) # don't include IC
            loss_rollout = torch.mean((Yi-Y_rollout)**2*mode_weights)
            loss_rollout2 = torch.mean((Yj-Y_rollout2)**2*mode_weights)
            loss = (loss+loss_rollout/rollout + loss_rollout2)/3*1e2
            print(epoch, 'loss: ', np.round(loss.detach().numpy(),6))
            loss.backward() # Does backpropagation and calculates gradients
            optimizer.step() # Updates the weights accordingly
            
    return model
    
