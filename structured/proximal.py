# Define a function that generates samples approximate RGO. The target is defined in Potential class.
import numpy as np
import targets
import random

# Define a function as the original approximate RGO
def generate_samples(step_size, x_y, f):
    dimension = f.dimension
    ite = 0
    while True:
        samples = np.random.multivariate_normal(mean = x_y, cov = step_size * np.identity(dimension), size = 2)
        # Compute the acceptance probability
        gradient = f.firstOrder(x_y)
        a = f.zeroOrder(samples[0,:])-np.dot(gradient,samples[0,:])
        b = f.zeroOrder(samples[1,:])-np.dot(gradient,samples[1,:])
        # The code works even when rho is inf. One can also take the log transformation
        rho = np.exp(b-a)
        u = np.random.uniform(0,1)
        ite = ite + 1
        if u < rho/2:
            break
    return samples[0,:],ite

# Define a function that estimates the local step size
def estimate_step_size(step_size, tolerance, y, f, fixed):
    dimension = f.dimension
    if fixed == True:
        return step_size, f.solve2(y, step_size)
    else:
        # Compute the desired subexponential parameter
        x_y = f.solve1(y, step_size)
        testFunction = lambda C : np.mean(np.exp(np.abs(Y)**(2/(1+f.alpha))/C))-2
        while True:
            Y = np.zeros(100)
            for i in range(100):
                # Generate random samples from a Gaussian distribution: \exp^{-(x-x_y)^2/(2\step_size)}
                samples = np.random.multivariate_normal(mean = x_y, cov = step_size * np.identity(dimension), size = 2)
                gradient = f.firstOrder(x_y)
                a = f.zeroOrder(samples[0])-np.dot(gradient,samples[0])
                b = f.zeroOrder(samples[1])-np.dot(gradient,samples[1])
                Y[i] = b-a
            # Estimate the subexponential parameter of Y: find the smallest C>0 such that E[\exp^{\abs(Y)/C}] \leq 2 by binary search for smooth potentials
            # Initialize the interval
            left = 0
            right = dimension**(f.alpha/(f.alpha+1)) # The estimated upper bound of the subexponential parameter
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
            if 10* mid < 1 / ( np.log(6/tolerance) / np.log(2) ):
                break
            else:
                step_size = step_size / 2
                x_y = f.solve1(y, step_size)
        return step_size, x_y

# Define the outer loop of proximal sampler
def proximal_sampler(initial_step_size, num_samples, num_iter, f, fixed):
    dimension = f.dimension
    stoppingiter = None
    samples = np.zeros([num_samples,num_iter,dimension])
    step_sizes = np.zeros([num_samples,num_iter])
    step_sizes[:,0] = initial_step_size
    ite_rejection = 0
    # Initialize the Ysamples
    initial_samples = np.random.multivariate_normal(mean = np.zeros(dimension), cov = np.identity(dimension), size = num_samples)
    Ysamples = np.zeros([num_samples,dimension])
    for j in range(num_samples):
        Ysamples[j,:] = np.random.multivariate_normal(mean = initial_samples[j,:], cov = initial_step_size * np.identity(dimension), size = 1)
    for i in range(num_iter):
        for j in range(num_samples):
            # Compute the new stationary point
            if i == 0:
                currentStepSize = initial_step_size
            else:
                currentStepSize = step_sizes[j,i-1]
            if fixed == False:
                if i > 0 and num_samples > 50:
                    if  np.var(step_sizes[:,i-1]/np.max(step_sizes[:,i-1])) < 0.01:
                        fixed = True
                        if stoppingiter == None:
                            stoppingiter = i+1
                if i >= 50 and num_samples <= 50:
                    sampleIndex = random.randrange(num_samples)
                    if  np.var(step_sizes[sampleIndex,i-50:i-1]/np.max(step_sizes[sampleIndex,i-50:i-1])) < 0.01:
                        fixed = True
                        if stoppingiter == None:
                            stoppingiter = i+1
            # if fixed is True at the beginning (fixed step size), fixed is always True
            if i % 10000 == 0 and i > 0:
                fixed = False
                step_size, x_y = estimate_step_size(initial_step_size, 1e-4, Ysamples[j,:], f, fixed)
            else:
                step_size, x_y = estimate_step_size(currentStepSize, 1e-4, Ysamples[j,:], f, fixed)
                # one can test if estimate_step_size() is robust with the following code
                # print(f"1:{step_size}")
                # step_size, x_y = estimate_step_size(currentStepSize, 1e-2, Ysamples[j,:], f, fixed)
                # print(f"2:{step_size}")
                # step_size, x_y = estimate_step_size(currentStepSize, 1e-2, Ysamples[j,:], f, fixed)
                # print(f"3:{step_size}")
            step_sizes[j,i] = step_size
            # When fixed is True, the following line is just solving the optimization problem.
            step_size, x_y = estimate_step_size(step_sizes[j,i], 1e-4, Ysamples[j,:], f, True)
            samples[j,i,:], ite = generate_samples(step_size, x_y, f)
            ite_rejection = ite_rejection + ite
            Ysamples[j,:] = np.random.multivariate_normal(mean = samples[j,i,:], cov = step_size * np.identity(dimension), size = 1)
            if i % 5000 == 0 and j == 0:
                print(i)
                print(f"step_size:{step_size}")
                if f.times2 > 0:
                    print(f"Averaged optimization steps of the new one: {f.ite2/f.times2}")
                print(f"Averaged rejection steps : {ite_rejection/(i+1)}")
    return samples, step_sizes, stoppingiter

# Define MALA
def MALA(Target, eta, num_iter, numSamplers):
    Xsamples = np.zeros([numSamplers, num_iter,Target.dimension])
    Xsamples[:, 0,:] = np.random.multivariate_normal(mean = np.zeros(Target.dimension), cov = np.identity(Target.dimension), size = numSamplers)
    for j in range(numSamplers):
        for i in range(num_iter-1):
            standard_Gaussian = np.random.multivariate_normal(mean = np.zeros(Target.dimension), cov = np.identity(Target.dimension), size = 1)
            x = Xsamples[j,i,:]
            y = x - eta*Target.firstOrder(x) + np.sqrt(2*eta)*standard_Gaussian
            tempxy = x - y + eta * Target.firstOrder(y)
            pxy = np.exp(-np.dot(tempxy, tempxy)/(4*eta))
            tempyx = y - x + eta * Target.firstOrder(x)
            pyx = np.exp(-np.dot(tempyx, tempyx)/(4*eta))
            r1 = Target.density(y)/Target.density(x)
            r2 = pxy/pyx
            acc_prob = min(1,r1*r2)
            U = np.random.uniform(0,1)
            if U <= acc_prob:
                x = y
            Xsamples[j,i+1,:] = x
    return Xsamples