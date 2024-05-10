# Define a function that generates samples from a one-dimentional Mixture Gaussian with approximate RGO 
# The target is \exp^{-(x-1)^2/2}+\exp^{-(x+1)^2/2}
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import random

# Define a function as the original approximate RGO
def generate_samples(step_size, x_y):
    f = potential()
    while True:
        # Generate random samples from a Gaussian distribution: \exp^{-(x-x_y)^2/(2\step_size)}
        samples = np.random.normal(x_y, np.sqrt(step_size), 2)
        # Compute the acceptance probability
        a = f.zero_order(samples[0])-f.first_order(x_y)*samples[0]
        b = f.zero_order(samples[1])-f.first_order(x_y)*samples[1]
        rho = np.exp(b-a)
        # Sample from a uniform distribution in [0,1]
        u = np.random.uniform(0,1)
        if u < rho/2:
            break
    return samples[0]

# Define a function that estimates the local step size
def estimate_step_size(step_size, tolerance, y):
    f = potential()
    return step_size, f.solve(y, step_size)
    # Compute the desired subexponential parameter
    desired_value = 1/(2*np.pi*np.log(16/(np.sqrt(3)*tolerance)) / np.log(2))
    x_y = f.solve(y, step_size)
    while True:
        Y = np.zeros(100)
        for i in range(100):
            # Generate random samples from a Gaussian distribution: \exp^{-(x-x_y)^2/(2\step_size)}
            samples = np.random.normal(x_y, np.sqrt(step_size), 2)
            a = f.zero_order(samples[0])-f.first_order(x_y)*samples[0]
            b = f.zero_order(samples[1])-f.first_order(x_y)*samples[1]
            Y[i] = b-a
        # Estimate the subexponential parameter of Y: find the smallest C>0 such that E[\exp^{\abs(Y)/C}] \leq 2 by binary search
        # Initialize the interval
        left = 0
        right = 1
        while np.mean(np.exp(np.abs(Y)/right))-2>0:
            left = right
            right = 2*right
        # Initialize the middle point
        mid = (left+right)/2
        # Initialize the value of the function
        f_mid = np.mean(np.exp(np.abs(Y)/mid))-2
        while abs(f_mid) > 1e-3:
            if f_mid > 0:
                left = mid
            else:
                right = mid
            mid = (left+right)/2
            f_mid = np.mean(np.exp(np.abs(Y)/mid))-2
        # Compute the estimated subexponential parameter
        estimated_value = 5*mid/np.pi
        if estimated_value  < desired_value:
            break
        else:
            step_size = step_size / 2
            x_y = f.solve(y, step_size)
    return step_size, x_y

# Define the outer loop of proximal sampler
def proximal_sampler(initial_step_size, num_samples, num_iter):
    f = potential()
    samples = np.zeros([num_samples,num_iter])
    step_sizes = np.zeros([num_samples,num_iter])
    step_sizes[:,0] = initial_step_size
    # Initialize the Ysamples
    # Ysamples = np.random.normal(0, np.sqrt(step_size), num_samples)
    Ysamples = np.zeros(num_samples)
    for i in range(1,num_iter):
        for j in range(num_samples):
            # Compute the new stationary point
            step_size, x_y = estimate_step_size(step_sizes[j,i-1], 1e-3, Ysamples[j])
            step_sizes[j,i] = step_size
            samples[j,i] = generate_samples(step_size, x_y)
            Ysamples[j] = np.random.normal(samples[j,i], np.sqrt(step_size), 1)
            print(i,j)
    # samples[:,0] = np.random.normal(0, np.sqrt(initial_step_size), num_samples)
    # samples[:,0] = np.zeros(num_samples)
    # Ysamples = np.zeros(num_samples)
    # for i in range((num_iter-1)):
    #     for j in range(num_samples):
    #         Ysamples[j] = np.random.normal(samples[j,i], np.sqrt(initial_step_size), 1)
    #         # Compute the new stationary point
    #         step_size, x_y = estimate_step_size(initial_step_size, 1e-3, Ysamples[j])
    #         samples[j,i+1] = generate_samples(step_size, x_y)
    #         print(i,j)
    return samples, step_sizes

# Define the potential function and its gradient
class potential(object):
    def zero_order(self, x):
        f_x = 1/2*(x-1)**2-np.log(1+np.exp(-2*x))
        return f_x
    def first_order(self, x):
        df_x = x-1+2/(1+np.exp(2*x))
        return df_x
    # Solve the equation df(x)+(1/eta)*(x-y)=0 with bisection search in the interval [0,y]
    def solve(self, y, eta):
        # Initialize the interval 
        left = min(0,y)
        right = max(0,y)
        # Initialize the middle point
        mid = (left+right)/2
        # Initialize the value of the function
        f_mid = self.first_order(mid)+(1/eta)*(mid-y)
        while abs(f_mid) > 1e-3:
            if f_mid > 0:
                right = mid
            else:
                left = mid
            mid = (left+right)/2
            f_mid = self.first_order(mid)+(1/eta)*(mid-y)
        return mid

np.random.seed(10)
num_iter = 2
Xsamples, step_sizes = proximal_sampler(0.01, num_samples = 1, num_iter = num_iter)
constant = 2 * np.sqrt(2 * np.pi)
distances = np.zeros(num_iter)
for i in range(num_iter):
    bin_num = 100
    hist, bin_edges = np.histogram(Xsamples[:,i], bins=bin_num, density=True)
    # Calculate the L1 distance estimate
    def l1_distance(x):
        f_x = (np.exp(-(x-1)**2/2)+np.exp(-(x+1)**2/2)) / constant
        index = np.searchsorted(bin_edges, x, side='left')
        if index == 0 or index == bin_num+1:
            s_x = 0
        else:
            s_x = hist[np.searchsorted(bin_edges, x, side='left')-1]
        return np.abs( s_x - f_x )
    # Test if s_x returns the density of samples distributon
    # def ssx(x):
    #     index = np.searchsorted(bin_edges, x, side='left')
    #     if index == 0 or index == bin_num+1:
    #         s_x = 0
    #     else:
    #         s_x = hist[np.searchsorted(bin_edges, x, side='left')-1]
    #     return s_x
    # f_values = [ssx(x) for x in np.linspace(-1,1,100)]
    # plt.plot(np.linspace(-1,1,100),f_values)
    # plt.hist(Xsamples[:,i], bins=bin_num, edgecolor='black',density=True)
    # plt.show()
    distances[i], _ = quad(l1_distance, -np.inf, np.inf)
step_sizes = np.delete(step_sizes,(0),1)
plt.plot(np.log(distances/2))
plt.show()
plt.plot(np.mean(step_sizes,axis=0))
plt.show()
plt.plot(np.var(step_sizes,axis=0))
plt.show()
plt.hist(Xsamples[:,num_iter-1], bins=100, edgecolor='black')
plt.show()
plt.plot(np.abs(np.mean(Xsamples,axis = 0)))
plt.show()
