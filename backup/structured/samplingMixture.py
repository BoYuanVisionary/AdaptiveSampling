# Define a function that generates samples approximate RGO. The target is defined in Potential class.
import numpy as np
import matplotlib.pyplot as plt
import targets
import cProfile
import pstats
import random
import argparse
import os
from utils import target_func, mixing_time


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
def estimate_step_size(step_size, tolerance, y, f):
    dimension = f.dimension
    # Compute the desired subexponential parameter
    x_y = f.solve1(y, step_size)
    testFunction = lambda C : np.mean(np.exp(np.abs(Y)**(2/(1+f.alpha))/C))-2
    while True:
        # Generate random samples from a Gaussian distribution: \exp^{-(x-x_y)^2/(2\step_size)}
        samples = np.random.multivariate_normal(mean = x_y, cov = step_size * np.identity(dimension), size = 200)
        Y = np.zeros(100)
        for i in range(100):
            gradient = f.firstOrder(x_y)
            a = f.zeroOrder(samples[i])-np.dot(gradient,samples[i])
            b = f.zeroOrder(samples[i+100])-np.dot(gradient,samples[i+100])
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
        while abs(f_mid) > 1e-2:
            if f_mid > 0:
                left = mid
            else:
                right = mid
            mid = (left+right)/2
            f_mid =  testFunction(mid)
        # reduce this 10
        if mid < 1 / ( np.log(6/tolerance) / np.log(2) ) * 10:
            break
        else:
            step_size = step_size / 2
            x_y = f.solve1(y, step_size)
    return step_size, x_y

# Define the outer loop of proximal sampler
def proximal_sampler(initial_step_size, num_samples, num_iter, f, fixed):
    dimension = f.dimension

    samples = np.zeros([num_samples,num_iter,dimension])
    Ysamples = np.zeros([num_samples,dimension])
    rejections = np.zeros([num_samples, num_iter])
    
    # Initialize the samples for both fxied and adaptive versions
    samples[:,0,:] = np.random.multivariate_normal(mean = np.zeros(dimension), cov = (1/f.L) * np.identity(dimension), size = num_samples)
    for j in range(num_samples):
        Ysamples[j,:] = np.random.multivariate_normal(mean = samples[j,0,:], cov = initial_step_size * np.identity(dimension), size = 1)
    
    if fixed == True:    
        print(f'fixed sampling')
        for i in range(1,num_iter):
            for j in range(num_samples):
                x_y = f.solve1(Ysamples[j,:], initial_step_size)
                samples[j,i,:],ite = generate_samples(initial_step_size, x_y, f)
                Ysamples[j,:] = np.random.multivariate_normal(mean = samples[j,i,:], cov = initial_step_size * np.identity(dimension), size = 1) 
                rejections[j,i] = ite
            if i % 50 == 0:
                print(f"Steps:{i}")
                if f.times2 > 0:
                    print(f"Averaged optimization steps of the new one: {f.ite2/f.times2}")
                print(f"Averaged rejection steps : {np.mean(rejections[:,i])}")
        
        return samples
    
    if fixed == False:
        print(f'adpative sampling')
        step_sizes = np.zeros([num_samples,num_iter])
        step_sizes[:,0] = initial_step_size
        for i in range(1,num_iter):
            for j in range(num_samples):
                # if i < 100 or (i >= 100 and np.random.uniform(0,1) < 0.001):
                if True:
                    step_size, x_y = estimate_step_size(step_sizes[j,i-1], 1e-2, Ysamples[j,:], f)
                else:
                    x_y = f.solve1(Ysamples[j,:], step_sizes[j,i-1])
                    step_size = step_sizes[j,i-1]
                samples[j,i,:],ite = generate_samples(step_size, x_y, f)
                Ysamples[j,:] = np.random.multivariate_normal(mean = samples[j,i,:], cov = step_size * np.identity(dimension), size = 1)  
                step_sizes[j,i] = step_size
                rejections[j,i] = ite  
            
            # statistics for the first sample
            if i % 50 == 0:
                print(f"Steps:{i}")
                print(f"Averaged_step_size:{np.mean(step_sizes[:,i])}")
                if f.times2 > 0:
                    print(f"Averaged optimization steps of the new one: {f.ite2/f.times2}")
                print(f"Averaged rejection steps : {np.mean(rejections[:,i])}")
            
        return samples, step_sizes
            # if fixed == False:
            #     if i > 0 and num_samples > 50:
            #         if  np.var(step_sizes[:,i-1]/np.max(step_sizes[:,i-1])) < 0.01:
            #             fixed = True
            #             if stoppingiter == None:
            #                 stoppingiter = i+1
            #     if i >= 50 and num_samples <= 50:
            #         sampleIndex = random.randrange(num_samples)
            #         if  np.var(step_sizes[sampleIndex,i-50:i-1]/np.max(step_sizes[sampleIndex,i-50:i-1])) < 0.01:
            #             fixed = True
            #             if stoppingiter == None:
            #                 stoppingiter = i+1
            # # if fixed is True at the beginning (fixed step size), fixed is always True
            # if i % 10000 == 0 and i > 0:
            #     fixed = False
            #     step_size, x_y = estimate_step_size(initial_step_size, 1e-2, Ysamples[j,:], f, fixed)
            # else:
            #     step_size, x_y = estimate_step_size(currentStepSize, 1e-2, Ysamples[j,:], f, fixed)
            #     # one can test if estimate_step_size() is robust with the following code
            #     # print(f"1:{step_size}")
            #     # step_size, x_y = estimate_step_size(currentStepSize, 1e-2, Ysamples[j,:], f, fixed)
            #     # print(f"2:{step_size}")
            #     # step_size, x_y = estimate_step_size(currentStepSize, 1e-2, Ysamples[j,:], f, fixed)
            #     # print(f"3:{step_size}")
            # step_sizes[j,i] = step_size
            # samples[j,i,:],ite = generate_samples(step_size, x_y, f)
            # Ysamples[j,:] = np.random.multivariate_normal(mean = samples[j,i,:], cov = step_size * np.identity(dimension), size = 1)

def MALA(Target, eta, num_iter, numSamplers):
    Xsamples = np.zeros([numSamplers, num_iter,Target.dimension])
    Xsamples[:, 0,:] = np.random.multivariate_normal(mean = np.zeros(Target.dimension), cov = np.identity(Target.dimension), size = numSamplers)
    for j in range(numSamplers):
        for i in range(num_iter-1):
            standard_Gaussian = np.random.multivariate_normal(mean = np.zeros(Target.dimension), cov = np.identity(Target.dimension), size = 1)
            x = Xsamples[j,i,:]
            y = x - eta*Target.firstOrder(x) + np.sqrt(2*eta)*standard_Gaussian
            y = y.ravel()
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


def sampler(seed, dimension, numIter, numSamples, initialStep):

    random.seed(seed)
    np.random.seed(seed)

    # direction = np.random.multivariate_normal(mean = np.zeros(dimension), cov =  np.identity(dimension), size = 1)
    # direction = np.transpose(direction) / np.linalg.norm(direction, ord = 2)
    # distance = np.random.multivariate_normal(mean = np.zeros(dimension), cov =  np.identity(dimension), size = 1)
    # distance = (distance[0,:]) / np.linalg.norm(distance[0,:], ord = 2)
    
    Target = target_func(dimension, if_simple = True)
    
    # To Do: double check this formula
    hatC = (1+Target.alpha)*(1/Target.alpha)**(Target.alpha/(1+Target.alpha))*(1/np.pi)**(2/(1+Target.alpha))*2**((-1-2*Target.alpha)/(1+Target.alpha))
    min_step_size = hatC/(120*np.log(6/0.01)/np.log(2)*Target.L_alpha*np.sqrt(Target.dimension))
    print(f"min_step_size={min_step_size}")
    
    # step size = 0.1 is good enough
    # RGO with fixed step size, the initial large one
    
    Xsamples_fixed_1 = proximal_sampler(1, numSamples, numIter, f = Target, fixed = True) 
    Xsamples_fixed_2 = proximal_sampler(0.1, numSamples, numIter, f = Target, fixed = True) 
    Xsamples_fixed_3 = proximal_sampler(0.01, numSamples, numIter, f = Target, fixed = True) 
    Xsamples_adaptive ,step_sizes = proximal_sampler(initialStep, numSamples, numIter, f = Target, fixed = False)
    
    plt.plot(np.mean(step_sizes,axis=0))
    plt.title('Mean of the step_size')
    plt.savefig('step_size.png')
    plt.close()
    
    direction = np.random.multivariate_normal(mean = np.zeros(dimension), cov =  np.identity(dimension), size = 1)
    direction = np.transpose(direction) / np.linalg.norm(direction, ord = 2)
    # 0. True density
    realSamples = np.zeros([200000, dimension])
    for i in range(np.shape(realSamples)[0]):
            realSamples[i,:] = Target.samplesTarget()
    bin_num = 500
    plt.hist(np.matmul(realSamples,direction), bins=bin_num, edgecolor='black', density=True, fill=False)
    plt.savefig(f"true_density.png")
    plt.close()

    mixing_time(Xsamples_fixed_1, 'fixed proximal sampler1', Target, direction, realSamples)
    mixing_time(Xsamples_fixed_2, 'fixed proximal sampler2', Target, direction, realSamples)
    mixing_time(Xsamples_fixed_3, 'fixed proximal sampler3', Target, direction, realSamples)
    mixing_time(Xsamples_adaptive, 'ada proximal sampler3', Target, direction, realSamples)

    
    # Xsamples_fixed = proximal_sampler(initialStep, numSamples, numIter, f = Target, fixed = True) 
    # np.save(f"fixed_samples_{dimension}_{numIter}_{numSamples}_{initialStep}.npy", Xsamples_fixed)
    # # RGO with adaptive step size
    # Xsamples_adaptive ,step_sizes = proximal_sampler(initialStep, numSamples, numIter, f = Target, fixed = False)
    # np.save(f"adaptive_samples_{dimension}_{numIter}_{numSamples}_{initialStep}.npy", Xsamples_adaptive)
    # plt.plot(np.mean(step_sizes,axis=0))
    # plt.title('Mean of the step_size')
    # plt.savefig('step_size.png')
    # plt.close()
    # # MALA
    # print('MALA')
    # Xsamples_MALA = MALA(Target, initialStep, numIter, numSamples)
    # np.save(f"MALA_samples_{dimension}_{numIter}_{numSamples}_{initialStep}.npy", Xsamples_MALA)

    # # uniform direction
    # direction = np.random.multivariate_normal(mean = np.zeros(dimension), cov =  np.identity(dimension), size = 1)
    # direction = np.transpose(direction) / np.linalg.norm(direction, ord = 2)
    # # 0. True density
    # realSamples = np.zeros([200000, dimension])
    # for i in range(np.shape(realSamples)[0]):
    #         realSamples[i,:] = Target.samplesTarget()
    # bin_num = 500
    # plt.hist(np.matmul(realSamples,direction), bins=bin_num, edgecolor='black', density=True, fill=False)
    # plt.savefig(f"true_density.png")
    # plt.close()
    
    # mixing_time(Xsamples_adaptive, 'adaptive proximal sampler', Target, direction, realSamples)
    # mixing_time(Xsamples_fixed, 'fixed proximal sampler', Target, direction, realSamples)
    # mixing_time(Xsamples_MALA, 'MALA', Target, direction, realSamples)

    # # Adaptive Sampler
    # Xsamples, stepSizes, stoppingIter = diagnosis(target = Target, numIter= numIter, initialStep = initialStep, numSamples = numSamples, fixed = False, direction=direction, distance=distance)
    # # Fixed one
    # # diagnosis(target = Target, numIter= numIter, initialStep = np.mean(stepSizes[:,-1]), numSamples = numSamples, fixed = True, direction=direction, distance=distance)
    # # print(f"stoppingIter:{stoppingIter}")
    if Target.times1 > 0:
        print(f"# of calls of built-in: {Target.times1}")
    if Target.times2 > 0:
        print(f"Averaged optimization steps of the new one: {Target.ite2/Target.times2}")
        print(f"# of calls of the new one: {Target.times2}")
    print(f"dimension:{dimension}")
    print(f"alpha:{Target.alpha}")
    # print(f"length:{Target.components}")
    # print(f"eigenvalues:{1/max(eigenvalues)} and {1/min(eigenvalues)}")
    
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Adaptive proximal samplers")
    parser.add_argument("--seed", type=int, default = 0)
    parser.add_argument("--dimension", type=int, default = 5)
    parser.add_argument("--iteration", type=int, default = 100)
    parser.add_argument("--samples", type=int, default = 100)
    parser.add_argument("--initialStep", type=float, default = 5)
    
    args = parser.parse_args()

    # Profile the main function
    cProfile.run("sampler(args.seed,args.dimension,args.iteration, args.samples, args.initialStep)", 
                sort="cumulative", filename="profile_results")
    stats = pstats.Stats("profile_results")
    # Print the top 30 results by cumulative time
    stats.sort_stats("cumulative").print_stats(30)

# more challenging target has a large K and hence more iterations for the optimization
