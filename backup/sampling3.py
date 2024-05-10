# Define a function that generates samples approximate RGO. The target is defined in Potential class.
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import minimize

# Define a function as the original approximate RGO
def generate_samples(step_size, x_y, f):
    dimension = f.dimension
    while True:
        # Generate random samples from a Gaussian distribution : \exp^{-(x-x_y)^2/(2\step_size)}
        samples = np.random.multivariate_normal(mean = x_y, cov = step_size * np.identity(dimension), size = 2)
        # Compute the acceptance probability
        a = f.zero_order(samples[0,:])-np.dot(f.first_order(x_y),samples[0,:])
        b = f.zero_order(samples[1,:])-np.dot(f.first_order(x_y),samples[1,:])
        # The code works even when rho is inf. One can also take the log transformation
        rho = np.exp(b-a)
        # Sample from a uniform distribution in [0,1]
        u = np.random.uniform(0,1)
        if u < rho/2:
            break
    return samples[0,:]

# Define a function that estimates the local step size
def estimate_step_size(step_size, tolerance, y, f, fixed):
    dimension = f.dimension
    if fixed == True:
        return step_size, f.solve(y, step_size)
    else:
        # Compute the desired subexponential parameter
        # desired_value = ( 1/(2*np.pi*np.log(16/(np.sqrt(3)*tolerance)) / np.log(2))) 
        x_y = f.solve(y, step_size)
        testFunction = lambda C : np.mean(np.exp(np.abs(Y)**(2/(1+f.alpha))/C))-2
        while True:
            Y = np.zeros(100)
            for i in range(100):
                # Generate random samples from a Gaussian distribution: \exp^{-(x-x_y)^2/(2\step_size)}
                samples = np.random.multivariate_normal(mean = x_y, cov = step_size * np.identity(dimension), size = 2)
                a = f.zero_order(samples[0])-np.dot(f.first_order(x_y),samples[0])
                b = f.zero_order(samples[1])-np.dot(f.first_order(x_y),samples[1])
                Y[i] = b-a
            # Estimate the subexponential parameter of Y: find the smallest C>0 such that E[\exp^{\abs(Y)/C}] \leq 2 by binary search
            # Initialize the interval
            left = 0
            right = dimension
            while testFunction(right)>0:
                left = right
                right = 2*right
            # Initialize the middle point
            mid = (left+right)/2
            # Initialize the value of the function
            f_mid = testFunction(mid)
            while abs(f_mid) > 1e-3:
                if f_mid > 0:
                    left = mid
                else:
                    right = mid
                mid = (left+right)/2
                f_mid =  testFunction(mid)
            # Compute the estimated subexponential parameter
            # estimated_value = mid/(np.pi*10)
            if 10* mid < 1 / ( np.log(3/tolerance) / np.log(2) + 1 ):
                break
            else:
                step_size = step_size / 2
                x_y = f.solve(y, step_size)
        return step_size, x_y

# Define the outer loop of proximal sampler
def proximal_sampler(initial_step_size, num_samples, num_iter, f, fixed):
    dimension = f.dimension
    samples = np.zeros([num_samples,num_iter,dimension])
    step_sizes = np.zeros([num_samples,num_iter])
    step_sizes[:,0] = initial_step_size
    # Initialize the Ysamples
    Ysamples = np.random.multivariate_normal(mean = np.zeros(dimension), cov =  0.01* np.identity(dimension), size = num_samples)
    for i in range(num_iter):
        for j in range(num_samples):
            # Compute the new stationary point
            if i == 0:
                currentStepSize = initial_step_size
            else:
                currentStepSize = step_sizes[j,i-1]
            step_size, x_y = estimate_step_size(currentStepSize, 1e-3, Ysamples[j,:], f, fixed)
            step_sizes[j,i] = step_size
            samples[j,i,:] = generate_samples(step_size, x_y, f)
            Ysamples[j,:] = np.random.multivariate_normal(mean = samples[j,i,:], cov = step_size * np.identity(dimension), size = 1)
            if j % 100 == 0:
                print(i,j)
    return samples, step_sizes

class TargetMixture(object):
    def __init__(self, dimension, delta, alpha):
        self.dimension = dimension
        self.delta = delta
        self.center = np.ones(self.dimension) / np.linalg.norm(np.ones(self.dimension), ord = 2)
        # The following three parameters should be pre-specified for new targets. For the adaptive version, the mean should be given.
        self.alpha = alpha
        self.L_alpha = 5
        self.mean = 0
        self.M = self.L_alpha**(2/(self.alpha+1))/((self.alpha+1)*delta)**((1-self.alpha)/(1+self.alpha))
        self.theta = (1-self.alpha)* delta /2

    def density(self, x):
        norm1 = np.linalg.norm(x-self.center, ord = 2)
        norm2 = np.linalg.norm(x+self.center, ord = 2)
        return np.exp(-norm1**(self.alpha+1) / 2) + np.exp(-norm2**(self.alpha+1) / 2)

    def zero_order(self, x):
        norm1 = np.linalg.norm(x-self.center, ord = 2)
        norm2 = np.linalg.norm(x+self.center, ord = 2)
        return 1/2 * norm2**(self.alpha+1) - np.log(1+np.exp(1/2*(norm2**(self.alpha+1)-norm1**(self.alpha+1))))

    def first_order(self, x):
        norm1 = np.linalg.norm(x-self.center, ord = 2)
        norm2 = np.linalg.norm(x+self.center, ord = 2)
        temp1 = (self.alpha+1)/2*(x-self.center)*norm1**(self.alpha-1)
        temp2 = (self.alpha+1)/2*(x+self.center)*norm2**(self.alpha-1)
        return temp1 - 1/(1+np.exp(1/2*(norm2**(self.alpha+1)-norm1**(self.alpha+1)))) * (temp1-temp2)

    # Solve the equation df(x)+(1/eta)*(x-y)=0 with accelerated gradient method
    def solve(self, input, eta):
        # This method is from 'A Proximal Algorithm for Sampling'. Does this also work for large eta?
        # x = np.zeros(self.dimension,1)
        # y = x
        # A = 0
        # tau = 1
        # m = 1/eta - self.L
        # M = 1/eta + self.L
        # g = lambda x : self.first_order(x)+ (x-input)/eta
        # while True:
        #     a = (tau + np.sqrt(tau**2 + 4*tau*M*A))/(2*M)
        #     ANext = A + a
        #     tx = A/ANext*y + a/ANext*x
        #     tauNext = tau + a*m
        #     yNext = tx - g(tx)/(m + M)
        #     xNext = (tau*x + a*m*tx - a*g(tx))/tauNext
        #     if np.linalg.norm(g(yNext), ord = 'fro')< np.sqrt(M*self.dimension):
        #         x_y = yNext
        #         break
        #     A = ANext
        #     x = xNext
        #     y = yNext
        #     tau = tauNext
        # return x_y
        # Standard optimization approach from scipy.optimize 
        targetFunction = lambda x: self.zero_order(x)+ 1/(2*eta)*np.linalg.norm(x-input, ord = 2)**2
        x0 = input
        # res = minimize(targetFunction, x0, method='nelder-mead', options={'fatol': 1e-4, 'disp': True, 'adaptive' : False})
        # res = minimize(targetFunction, x0, method='Powell', options={'disp': False})
        # The running speed is fast enough. The requirment for the gradient norm should be specified later. 
        res = minimize(targetFunction, x0, method='CG', options={'gtol': np.sqrt(self.dimension)*0.01, 'disp': False})
        print(np.linalg.norm( self.first_order(res.x)+(res.x-input)/eta, ord = 2)/np.sqrt(self.dimension))
        return res.x
    
    def samplesOneComponent(self, sign):
        centerNew = sign * self.center
        if self.alpha == 1:
            sample = np.random.multivariate_normal(mean = centerNew, cov =  np.identity(self.dimension), size = 1)
        else:
            while True:
                sample = np.random.multivariate_normal(mean = centerNew, 
                                                       cov =  np.sqrt(1/self.M) * np.identity(self.dimension), size = 1)
                norm = np.linalg.norm(sample-centerNew, ord = 2)
                logThres = -1/2 * norm **(self.alpha+1) - self.M/2 * norm ** 2 -self.theta
                if logThres > 0:
                    raise ValueError('The logThres is positive')
                if np.log(np.random.uniform(0,1)) < logThres:
                    break
        return sample

    def samplesTarget(self):
        if np.random.uniform(0,1) < 0.5:
            sample = self.samplesOneComponent(1)
        else:
            sample = self.samplesOneComponent(-1)
        return sample

def diagnosis(numIter, initialStep, numSamples, dimension, alpha, fixed):
    # numSamples >= 5; 0 <= alpha <= 1
    numIter = numIter
    initialStep = initialStep
    numSamples = numSamples

    f = TargetMixture(dimension = dimension, delta = 1, alpha = alpha)
    Xsamples, step_sizes = proximal_sampler(initialStep, numSamples, numIter, f = f, fixed = fixed)
    step_sizes = np.delete(step_sizes,0, axis=1)

    # 1. Discrete TV on the direction -center -> center'
    # generate samples from the mixture of two Gaussian distributions
    realSamples = np.zeros([1000, f.dimension])
    for i in range(1000):
            realSamples[i,:] = f.samplesTarget()
            print(i)
    # project samples onto one direction
    # direction = f.center / np.linalg.norm(f.center, ord = 2)
    direction = np.random.multivariate_normal(mean = np.zeros(f.dimension), cov =  np.identity(dimension), size = 1)
    direction = direction / np.linalg.norm(direction, ord = 2)
    bin_num = 100
    histY, bin_edges_Y = np.histogram(np.matmul(realSamples,direction), bins=bin_num, density=True)
    plt.hist(np.matmul(realSamples,direction), bins=100, edgecolor='black', density=True)
    plt.show()
    distances = np.zeros(numIter)
    for i in range(numIter):
        histX, bin_edges_X = np.histogram(np.matmul(Xsamples[:,i,:],direction), bins=bin_num, density=True)
        # Calculate the L1 distance estimate
        def l1_distance(x):
            index = np.searchsorted(bin_edges_X, x, side='left')
            if index == 0 or index == bin_num+1:
                s_x = 0
            else:
                s_x = histX[np.searchsorted(bin_edges_X, x, side='left')-1]
            
            index = np.searchsorted(bin_edges_Y, x, side='left')
            if index == 0 or index == bin_num+1:
                f_x = 0
            else:
                f_x = histY[np.searchsorted(bin_edges_Y, x, side='left')-1]
            return np.abs( s_x - f_x )
        
        distances[i], _ = quad(l1_distance, -np.inf, np.inf)
    # constant = 2 * np.sqrt(2 * np.pi)
    # distances = np.zeros(numIter)
    # for i in range(numIter):
    #     bin_num = 100
    #     hist, bin_edges = np.histogram(Xsamples[:,i,:], bins=bin_num, density=True)
    #     # Calculate the L1 distance estimate
    #     def l1_distance(x):
    #         f_x = (np.exp(-(x-1)**2/2)+np.exp(-(x+1)**2/2)) / constant
    #         index = np.searchsorted(bin_edges, x, side='left')
    #         if index == 0 or index == bin_num+1:
    #             s_x = 0
    #         else:
    #             s_x = hist[np.searchsorted(bin_edges, x, side='left')-1]
    #         return np.abs( s_x - f_x )
    #     distances[i], _ = quad(l1_distance, -np.inf, np.inf)

    plt.plot(np.log(distances/2))
    plt.title('discrete TV on the direction -center -> center')
    plt.show()

    # 2. Errors of the mean measured by L_2 norm /sqrt(dimension) 
    means = np.zeros(numIter)
    for i in range(numIter):
        means[i]  = np.linalg.norm( np.mean(Xsamples[:,i,:], axis = 0)- f.mean, ord = 2) / np.sqrt(f.dimension)
    plt.plot(means)
    plt.title('Errors of the mean measured by L_2 norm /sqrt(dimension) ')
    plt.show()

    # 3. Trace Plot
    plt.plot(np.transpose(Xsamples[0:1,:,0]))
    plt.title('Trace Plot on the 1st coordinate')
    plt.show()

    # 4. Mean of the step size
    plt.plot(np.arange(1,numIter),np.mean(step_sizes,axis=0))
    plt.title('Mean of the step_size')
    plt.show()

    # 5. Histogram of the last iteration
    plt.hist(np.matmul(Xsamples[:,numIter-1,:],direction), bins=100, edgecolor='black', density=True)
    plt.show()

    return Xsamples, step_sizes


Xsamples,stepSizes = diagnosis(numIter= 100, initialStep = 100, numSamples = 100, dimension = 5, alpha = 0.5, fixed = False)
diagnosis(numIter= 100, initialStep = 0.005, numSamples = 100, dimension = 5, alpha = 0.5, fixed = True)





