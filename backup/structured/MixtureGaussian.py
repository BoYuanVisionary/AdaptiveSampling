# Define a function that generates samples approximate RGO. The target is defined in Potential class.
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import targets
import cProfile
import pstats
import time
import random
import argparse
import os


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
    Ysamples = np.random.multivariate_normal(mean = np.zeros(dimension), cov = np.identity(dimension), size = num_samples)
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
            samples[j,i,:],ite = generate_samples(step_size, x_y, f)
            ite_rejection = ite_rejection + ite
            Ysamples[j,:] = np.random.multivariate_normal(mean = samples[j,i,:], cov = step_size * np.identity(dimension), size = 1)
            if i % 5000 == 0 and j == 0:
                print(i)
                print(f"step_size:{step_size}")
                if f.times2 > 0:
                    print(f"Averaged optimization steps of the new one: {f.ite2/f.times2}")
                print(f"Averaged rejection steps : {ite_rejection/(i+1)}")
    return samples, step_sizes, stoppingiter

def diagnosis(target, numIter, initialStep, numSamples, fixed, direction, distance):
    numIter = numIter
    initialStep = initialStep
    numSamples = numSamples

    f = target
    Xsamples, step_sizes, stoppingIter = proximal_sampler(initialStep, numSamples, numIter, f = f, fixed = fixed)
    step_sizes = np.delete(step_sizes,0, axis=1)

    number_samples_hist = int(3e2)
    min_number_samples_hist = int(1e2)
    number_hist = 100
    # Not the precise bin_size but it doesn't matter
    bin_size = int((number_samples_hist-min_number_samples_hist) / 100 )
    index_hist = np.linspace(min_number_samples_hist, number_samples_hist, number_hist)
    index_hist = index_hist.astype(int)

    # 0. True density
    realSamples = np.zeros([100000, f.dimension])
    for i in range(100000):
            realSamples[i,:] = f.samplesTarget()
    bin_num = 500


    # 1.1 Projection along 'direction'
    if numIter > number_samples_hist:
        projected_sample = np.matmul(realSamples,direction)
        plt.hist(projected_sample, bins=bin_num, edgecolor='black', density=True, fill=False)
        histY, bin_edges_Y = np.histogram(projected_sample, bins=bin_num, density=True)
        plt.hist(np.matmul(Xsamples[0,numIter-number_samples_hist-1:numIter-1,:],direction), bins=bin_num, edgecolor='red', density=True,fill=False)
        plt.show()
        distances_random = np.zeros(number_hist)
        projected_sample_random = np.matmul(Xsamples[0,numIter-number_samples_hist-1:numIter-1,:],direction)
        for i in range(number_hist):
            distances_random[i] = TV_estimation(projected_sample_random[-index_hist[i]:], histY, bin_edges_Y, bin_num)
        plt.plot(0.5*distances_random)
        plt.title(f"TV along direction, bin={bin_size}")
        plt.show()

    # 1.2 Projection along 'distance'
    if numIter > number_samples_hist:
        projected_sample = np.matmul(realSamples,distance)
        plt.hist(projected_sample, bins=bin_num, edgecolor='black', density=True, fill=False)
        histY, bin_edges_Y = np.histogram(projected_sample, bins=bin_num, density=True)
        plt.hist(np.matmul(Xsamples[0,numIter-number_samples_hist-1:numIter-1,:],distance), bins=bin_num, edgecolor='red', density=True,fill=False)
        plt.show()
        distances_random = np.zeros(number_hist)
        projected_sample_random = np.matmul(Xsamples[0,numIter-number_samples_hist-1:numIter-1,:],distance)
        for i in range(number_hist):
            distances_random[i] = TV_estimation(projected_sample_random[-index_hist[i]:], histY, bin_edges_Y, bin_num)
        plt.plot(0.5*distances_random)
        plt.title(f"TV along distance, bin={bin_size}")
        plt.show()
    
    # 1.3 Projection along 'orthogonal'
    random_vector = np.random.rand(f.dimension)
    dot_product = np.dot(distance, random_vector)
    orthogonal = random_vector - dot_product * distance
    if numIter > number_samples_hist:
        projected_sample = np.matmul(realSamples,orthogonal)
        plt.hist(projected_sample, bins=bin_num, edgecolor='black', density=True, fill=False)
        histY, bin_edges_Y = np.histogram(projected_sample, bins=bin_num, density=True)
        plt.hist(np.matmul(Xsamples[0,numIter-number_samples_hist-1:numIter-1,:],orthogonal), bins=bin_num, edgecolor='red', density=True,fill=False)
        plt.show()
        distances_random = np.zeros(number_hist)
        projected_sample_random = np.matmul(Xsamples[0,numIter-number_samples_hist-1:numIter-1,:],orthogonal)
        for i in range(number_hist):
            distances_random[i] = TV_estimation(projected_sample_random[-index_hist[i]:], histY, bin_edges_Y, bin_num)
        plt.plot(0.5*distances_random)
        plt.title(f"TV along orthogonal, bin={bin_size}")
        plt.show()

    # # constant = 2 * np.sqrt(2 * np.pi)
    # # distances = np.zeros(numIter)
    # # for i in range(numIter):
    # #     bin_num = 100
    # #     hist, bin_edges = np.histogram(Xsamples[:,i,:], bins=bin_num, density=True)
    # #     # Calculate the L1 distance estimate
    # #     def l1_distance(x):
    # #         f_x = (np.exp(-(x-1)**2/2)+np.exp(-(x+1)**2/2)) / constant
    # #         index = np.searchsorted(bin_edges, x, side='left')
    # #         if index == 0 or index == bin_num+1:
    # #             s_x = 0
    # #         else:
    # #             s_x = hist[np.searchsorted(bin_edges, x, side='left')-1]
    # #         return np.abs( s_x - f_x )
    # #     distances[i], _ = quad(l1_distance, -np.inf, np.inf)

    # plt.plot(np.log(distances/2))
    # plt.title('discrete TV on the direction -center -> center')
    # plt.show()

    # 2. Errors of the mean measured by L_2 norm /sqrt(dimension). 
    if numIter > 100000:
        means = np.zeros(numIter-100000)
        for i in range(100000,numIter):
            means[i-100000]  = np.linalg.norm( np.mean(Xsamples[:,i-100000:i,:])- f.mean, ord = 2) / np.sqrt(f.dimension)
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

    # 5. Histogram at the last iteration
    if numSamples > 100 :
        plt.hist(np.matmul(Xsamples[:,numIter-1,:],direction), bins=100, edgecolor='black', density=True)
        plt.show()
    return Xsamples, step_sizes, stoppingIter

# Discrete TV on a random direction 
def TV_estimation(sampleX, histY, bin_edges_Y, bin_num):

    histX, bin_edges_X = np.histogram(sampleX, bins=bin_num, density=True)
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
    bins = np.unique(np.concatenate((bin_edges_X,bin_edges_Y)))
    lower_bound = np.min(bins)
    upper_bound = np.max(bins)
        
    distance, _ = quad(l1_distance, lower_bound, upper_bound, epsabs = 1e-3, points = bins, limit = (bin_num+1)*2)
    return distance

if __name__ == "__main__":
    def main(seed,dimension,large_condition):

        random.seed(seed)
        np.random.seed(seed)

        numIter = int(5e2)
        numSamples = 1

        direction = np.random.multivariate_normal(mean = np.zeros(dimension), cov =  np.identity(dimension), size = 1)
        direction = np.transpose(direction) / np.linalg.norm(direction, ord = 2)

        # Define a Guassian Mixture distribution
        distance = np.random.multivariate_normal(mean = np.zeros(dimension), cov =  np.identity(dimension), size = 1)
        distance = 0.5* (distance[0,:]) / np.linalg.norm(distance[0,:], ord = 2)
        t1 = targets.SemiGaussianTarget(dimension = dimension, mean = distance, inverse = np.sqrt(1/2)* np.identity(dimension), alpha = 1, prob = 0.5)
        t2 = targets.SemiGaussianTarget(dimension = dimension, mean = -distance, inverse = np.sqrt(1/2)* np.identity(dimension), alpha = 1, prob = 0.5)
        Target = targets.TargetMixture(t1,t2)

        hatC = (1+Target.alpha)*(1/Target.alpha)**(Target.alpha/(1+Target.alpha))*(1/np.pi)**(2/(1+Target.alpha))*2**((-1-2*Target.alpha)/(1+Target.alpha))
        min_step_size = hatC/(120*np.log(6/0.01)/np.log(2)*Target.L_alpha*np.sqrt(Target.dimension))
        print(f"min_step_size={min_step_size}")

        # Adaptive one
        Xsamples, stepSizes, stoppingIter = diagnosis(target = Target, numIter= numIter, initialStep = 10, numSamples = numSamples, fixed = False, direction=direction, distance=distance)
        # Fixed one
        # diagnosis(target = Target, numIter= numIter, initialStep = np.mean(stepSizes[:,-1]), numSamples = numSamples, fixed = True, direction=direction, distance=distance)
        
        print(f"stoppingIter:{stoppingIter}")
        if Target.times1 > 0:
            print(f"# of calls of built-in: {Target.times1}")
        if Target.times2 > 0:
            print(f"Averaged optimization steps of the new one: {Target.ite2/Target.times2}")
            print(f"# of calls of the new one: {Target.times2}")
        print(f"dimension:{dimension}")
        print(f"alpha:{Target.alpha}")
        print(f"length:{Target.components}")
        # eigenvalues, _ = np.linalg.eig(specialInverse * np.transpose(specialInverse))
        # print(f"eigenvalues:{1/max(eigenvalues)} and {1/min(eigenvalues)}")
        return Xsamples,direction

    def generateInvertibleMatrix(dimension):
        while True:
            random_matrix = np.random.rand(dimension, dimension)
            if np.linalg.matrix_rank(random_matrix) == dimension:
                return random_matrix

    def generateInvertibleMatrixConditional(dimension,large_condition):
        # Generate a random matrix
        random_matrix = np.random.rand(dimension, dimension)
        random_orthogonal_matrix, _ = np.linalg.qr(random_matrix)
        if large_condition == True:
            temp = np.matmul(random_orthogonal_matrix, np.diag(np.sqrt(np.linspace(0.05,1,num=dimension))))
        else:
            temp = np.matmul(random_orthogonal_matrix, np.sqrt(0.1)*np.identity(dimension))
        output = np.matmul(temp,np.transpose(random_orthogonal_matrix))
        eigenvalues, _ = np.linalg.eig(np.matmul(output,np.transpose(output)))
        print(1/eigenvalues)
        return output

    # Test if the estimation of Lalpha is stable
    def stabilityLalpha():
        dimension = 50
        specialMean = np.random.multivariate_normal(mean = np.zeros(dimension), cov = np.identity(dimension), size = 3)
        specialInverse = np.identity(dimension)
        t1 = targets.SemiGaussianTarget(dimension = dimension, mean =  specialMean[2,:], inverse =  generateInvertibleMatrix(dimension), alpha = np.random.uniform(0,1) , prob = 0.5)
        t2 = targets.SemiGaussianTarget(dimension = dimension, mean =  specialMean[1,:], inverse =  generateInvertibleMatrix(dimension), alpha = 1 , prob = 0.5)
        Target = targets.TargetMixture(t1,t2)
        result = np.zeros(5)
        for i in range(5):
            result[i] = Target.estimation(10000)
        print(result)
    
    def testInequality():
        alpha = random.uniform(0,1)
        dimension = 10
        samples = np.random.multivariate_normal(mean = np.zeros(dimension), cov = np.identity(dimension), size = 200)
        for i in range(100):
            N1 = np.linalg.norm(samples[i,:], ord = 2)
            N2 = np.linalg.norm(samples[i+100,:], ord = 2)
            L = np.linalg.norm(samples[i,:] / N1**(1-alpha)-samples[i+100,:] / N2**(1-alpha), ord = 2)
            R = 2**(1-alpha) * np.linalg.norm(samples[i,:]-samples[i+100,:], ord = 2)**alpha
            print(R-L)

    parser = argparse.ArgumentParser(description="Adaptive proximal samplers")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dimension", type=int)
    parser.add_argument("--large_condition", action="store_true")
    parser.add_argument("--save_images", type=str, help="Folder to save images")
    args = parser.parse_args()

    seed = args.seed
    dimension = args.dimension
    large_condition = args.large_condition
    # Profile the main function
    cProfile.run("samples, direction = main(seed,dimension,large_condition)", sort="cumulative", filename="profile_results")
    stats = pstats.Stats("profile_results")
    # Print the top 30 results by cumulative time
    stats.sort_stats("cumulative").print_stats(30)


