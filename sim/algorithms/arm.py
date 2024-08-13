
# add necessary modules
import numpy as np
from scipy.stats import norm
import GPy
import random

# define variance of noise in observations
noisevar = 0.01

# Define the function that generates the data, get 0 at 0 and the maximum value 1 at 20, which is convenient for scaling
def linear_func(z):
    return z/20.0
def sigmoid_function(z):
    return 1 / (1 + np.exp(-(z - 10)))
def power_function(z):
    return z ** 2 / 400.0


# define arm class
class Arm:
    def __init__(self, arm_id, zinit, Zmax):
        self.arm_id = arm_id
        self.noisevar = noisevar
        self.zinit = zinit
        self.z = zinit
        self.Zmax = Zmax
        self.dataratio = 0.0
        self.datasize = 0
        self.datatype = 0
        # self.datatype = random.choice([0,1,2])
        self.noise = 0.
        # self.noise = random.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        self.gradient_quality = 0
        self.numplays = 0.0
        self.grad_hist = np.array([], dtype = np.int64)
        self.grad = 0.0
        self.zhist = np.array([], dtype = np.int64).reshape(0,1)
        self.yhist = np.array([], dtype = np.float64).reshape(0,1)
        self.div = 0.0
        self.ucb = 0.0
        self.qual = 0.0
        self.model = None
        self.kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=2.5)
        
        
    def UpdatePosterior(self, znew, ynew, grad_quality=10):
        self.zhist = np.vstack([self.zhist, znew])
        self.yhist = np.vstack([self.yhist, ynew])
        if grad_quality != 10:
            self.grad_hist = np.append(self.grad_hist, grad_quality)
        self.numplays += 1
        if self.model is None:
            self.model = GPy.models.GPRegression(X=self.zhist, Y=self.yhist,
                                                 kernel=self.kernel, noise_var = self.noisevar)
        else:
            self.model.set_XY(self.zhist, self.yhist)
        #  Optimizing model hyperparameters
        self.model.optimize(messages=True)

    # datatype_dict {0: linear_func, 1: sigmoid_function, 2: power_function}
    def datafunct(self, z):
        if self.datatype == 0:
            y = linear_func(z*20.0/self.Zmax)
        elif self.datatype == 1:
            y = sigmoid_function(z*20.0/self.Zmax)
        elif self.datatype == 2:
            y = power_function(z*20.0/self.Zmax)
        return y
    
    def reset(self):
        self.z = self.zinit
        self.numplays = 0.0
        self.zhist = np.array([], dtype = np.int64).reshape(0,1)
        self.yhist = np.array([], dtype = np.int64).reshape(0,1)
        self.ucb = 0.0
        self.model = None
        

