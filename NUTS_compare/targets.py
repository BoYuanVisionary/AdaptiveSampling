import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.special import gamma
import math
import random
# TargeMixture class works for any components not just components defined as \exp(-\|\Sigma^{-1}(x-mu)\|^(\alpha+1))
class TargetMixture(object):
    
    def __init__(self,*args):
        self.probs = np.zeros(len(args))
        alphas = np.zeros(len(args))
        integrals = np.zeros(len(args))
        dimensions = np.zeros(len(args))

        for i, item in enumerate(args):
            self.probs[i] = item.prob
            alphas[i] = item.alpha
            integrals[i] = item.integral
            dimensions[i] = item.dimension
        assert math.isclose(len(set(dimensions)), 1.0, rel_tol=1e-9) 
        self.dimension = int(dimensions[0])
        assert math.isclose(sum(self.probs), 1.0, rel_tol=1e-9)
        
        self.means = np.zeros([int(self.dimension),len(args)])
        for i, item in enumerate(args):
            self.means[:,i] = item.mean
        self.mean = np.matmul(self.means,self.probs) / np.inner(self.probs, integrals)
        # Take the maximal alphas of all components as the parameter of the mixtured one
        self.alpha = np.max(alphas)
        self.args = args
        self.components = len(args)
        # delta is used in Jiaming's optimization code
        delta = 1
        # Estimation of L_alpha of the mixtured measure
        self.L_alpha = self.estimation()
        self.L = self.L_alpha**(2/(self.alpha+1))/((self.alpha+1)*delta)**((1-self.alpha)/(1+self.alpha))
        self.times1 = 0
        self.ite2 = 0
        self.times2 = 0

    # return the density at x
    def density(self, x):
        density = 0
        for i, item in enumerate(self.args):
            density = density + self.probs[i] * item.density(x)
        return density

    # return the potential at x
    def zeroOrder(self, x):
        zeroOrders = np.zeros(self.components)
        for i, item in enumerate(self.args):
            zeroOrders[i] = -item.zeroOrder(x) + np.log(self.probs[i])
        return -logsumexp(zeroOrders)

    # return the gradient of the potential at x
    def firstOrder(self, x):
        parameters = np.zeros(self.components)
        zeroOrders = np.zeros(self.components)
        firstOrders = np.zeros([self.components,self.dimension])
        for i, item in enumerate(self.args):
            firstOrders[i,:] = item.firstOrder(x)
            zeroOrders[i] = item.zeroOrder(x) 
        parameters = np.exp(np.log(self.probs) - zeroOrders + self.zeroOrder(x))
        return np.matmul(np.transpose(parameters),firstOrders)

    # return a sample follwoing the mixtured measure
    def samplesTarget(self):
        chosenDis = random.choices(range(self.components), self.probs)[0]
        # print(chosenDis)
        return self.args[chosenDis].generateSample()

    # Solve the equation df(x)+(1/eta)*(x-y)=0 
    def solve1(self, input, eta):
        self.times1 = self.times1 + 1
        # Standard optimization approach from scipy.optimize 
        error = self.dimension**(1/(2*(self.alpha+1))) / (7*self.L_alpha**(1/(self.alpha+1))*eta)
        # Using dot instead of linglg.norm would be faster. For the squared L_2 norm, the improvment is more significant ~50%
        targetFunction = lambda x: self.zeroOrder(x)+ 1/(2*eta)*np.dot(x-input,x-input)
        x0 = input
        # res = minimize(targetFunction, x0, method='nelder-mead', options={'fatol': 1e-4, 'disp': True, 'adaptive' : False})
        # res = minimize(targetFunction, x0, method='Powell', options={'disp': False})
        res = minimize(targetFunction, x0, method='L-BFGS-B', options={'gtol': 1e-2, 'disp': False, 'maxiter': 10})
        if np.random.uniform(0,1) < 1/1000:
            print(np.linalg.norm( self.firstOrder(res.x)+(res.x-input)/eta, ord = 2)/np.sqrt(self.dimension))
        return res.x

    # This method is from 'A Proximal Algorithm for Sampling'. (Jiaming's implementation)
    def solve2(self, input, eta):
        error = self.dimension**(1/(2*(self.alpha+1))) / (7*self.L_alpha**(1/(self.alpha+1))*eta)
        x = np.zeros(self.dimension)
        y = x
        A = 0
        tau = 1
        m = 1/eta - self.L
        if m < 0:
            return self.solve1(input,eta)
        M = 1/eta + self.L
        g = lambda x : self.firstOrder(x)+ (x-input)/eta
        ite = 0
        while True:
            ite = ite + 1
            a = (tau + np.sqrt(tau**2 + 4*tau*M*A))/(2*M)
            ANext = A + a
            tx = A/ANext*y + a/ANext*x
            tauNext = tau + a*m
            yNext = tx - g(tx)/(m + M)
            xNext = (tau*x + a*m*tx - a*g(tx))/tauNext
            tempGradient = g(yNext)
            if np.dot(tempGradient, tempGradient) / self.dimension < 1e-4:
                x_y = yNext
                break
            if ite >= 5:
                x_y = yNext
                break
            # # If this takes too many iterations, use solve1()
            # if ite > 100:
            #     return self.solve1(input,eta)
            A = ANext
            x = xNext
            y = yNext
            tau = tauNext
        if np.random.uniform(0,1) < 1/1000:
            print(np.linalg.norm(g(x_y), ord = 2)/np.sqrt(self.dimension))
        self.ite2 = self.ite2 + ite 
        self.times2 = self.times2 + 1
        return x_y

    def estimation(self, times = 10000):
        L = 0
        for j in range(self.components):
            for i in range(times):
                samples = np.random.multivariate_normal(mean = self.means[:,j], cov = np.identity(self.dimension), size = 2)
                d1 = np.linalg.norm(self.firstOrder(samples[0,:])-self.firstOrder(samples[1,:]), ord = 2)
                d2 = np.linalg.norm(samples[0,:]-samples[1,:], ord = 2)**self.alpha
                L = max(L,d1/d2)
        print(f"L={L}")
        return L
    

class SemiGaussianTarget(object):

    def __init__(self, dimension, mean, inverse, alpha, prob):
        self.dimension = dimension
        self.mean = mean
        self.alpha = alpha
        self.prob = prob
        self.inverse = inverse
        self.cov = np.linalg.inv(inverse) 
        self.integral = self.dimension / (alpha+1) * np.linalg.det(inverse)**(-1) * np.pi**(dimension/2) \
            * gamma(dimension/(alpha+1)) / gamma(dimension/2+1)
        self.L_alpha = 0

    def density(self, x):
        temp_vector = np.matmul(self.inverse, x-self.mean)
        norm = np.sqrt(np.dot(temp_vector,temp_vector))
        return np.exp(- 0.5 * norm**(self.alpha+1))
    
    def zeroOrder(self, x):
        temp_vector = np.matmul(self.inverse, x-self.mean)
        norm = np.sqrt(np.dot(temp_vector,temp_vector))
        return  0.5 * norm**(self.alpha+1)

    def firstOrder(self, x):
        temp_vector = np.matmul(self.inverse, x-self.mean)
        norm = np.sqrt(np.dot(temp_vector,temp_vector))
        return 0.5 * (self.alpha+1) * np.matmul(np.matmul(np.transpose(self.inverse), self.inverse), (x-self.mean)) * norm**(self.alpha-1)

    def generateSample(self):
        v = np.random.gamma(shape = self.dimension / (self.alpha+1), scale = 1.0, size = 1)
        r = v ** (1/(self.alpha+1))
        direction = np.random.multivariate_normal(mean = np.zeros(self.dimension), cov =  np.identity(self.dimension), size = 1)
        direction = direction / np.linalg.norm(direction, ord = 2)
        sample = np.matmul(r*direction,self.cov ) + self.mean
        return sample


class NFarget(object):
    # We assume alpha is 1
    def __init__(self):
        self.dimension = 2
        self.alpha = 1
        self.prob = 1
        # not important
        self.mean = 0
        self.integral = 0

    def density(self, x):
        return np.exp(-self.zeroOrder(x))
    
    def zeroOrder(self, x):
        X = x[1]
        Y = x[0]
        norm = np.dot(X,X)
        return 1/6*Y**2 +  1/2*np.exp(-Y/2)*(X**2)

    def firstOrder(self, x):
        gradient = np.zeros(2)
        X = x[1]
        Y = x[0]
        norm = X**2
        gradient[0] = Y/3-norm/4*np.exp(-Y/2)
        gradient[1] = np.exp(-Y/2) * X
        return gradient
        
    def generateSample(self):
        sample = np.zeros(2)
        sample[0] = np.random.normal(0,np.sqrt(3))
        sample[1] = np.random.normal(0, np.sqrt(np.exp(sample[0]/2)))
        return sample