import numpy as np
import targets 
import matplotlib.pyplot as plt
from scipy.integrate import quad

def generateInvertibleMatrixConditional(dimension,large_condition=True):
    # Generate a random matrix
    random_matrix = np.random.rand(dimension, dimension)
    random_orthogonal_matrix, _ = np.linalg.qr(random_matrix)
    if large_condition == True:
        temp = np.matmul(random_orthogonal_matrix, np.diag(np.sqrt(np.linspace(1,100,num=dimension)))) # the eigenvalues are 0.01 to 1
    else:
        temp = np.matmul(random_orthogonal_matrix, np.sqrt(1)*np.identity(dimension)) # the eigenvalues are 1
    output = np.matmul(temp,np.transpose(random_orthogonal_matrix))
    eigenvalues, _ = np.linalg.eig(np.matmul(output,np.transpose(output)))
    print(1/eigenvalues)
    return output

def target_Gaussian(dimension):
    
    t1 = targets.SemiGaussianTarget(dimension = dimension, mean = np.zeros(dimension), inverse = np.identity(dimension), alpha = 1 , prob = 1)
    Target = targets.TargetMixture(t1)

    return Target

def target_MixedGaussian(dimension):   
    
    # a = np.random.multivariate_normal(mean = np.zeros(dimension), cov =  np.identity(dimension), size = 1)
    # a = (np.transpose(a) / np.linalg.norm(a, ord = 2)) /2
    a = np.zeros(dimension)
    a[0] = 2
    # a = np.array([2,0])
    t1 = targets.SemiGaussianTarget(dimension = dimension, mean = a, inverse = np.identity(dimension), alpha = 1 , prob = 1/2)
    t2 = targets.SemiGaussianTarget(dimension = dimension, mean = -a, inverse = np.identity(dimension), alpha = 1 , prob = 1/2)
    Target = targets.TargetMixture(t1,t2)
    
    return Target, a

def target_banana(dimension): # to do
    
    t1 = targets.NFarget(dimension)
    return Target

def target_funnel(dimension):
    
    t = targets.NFarget(dimension)
    return targets.TargetMixture(t)

def mixing_time(samples, sampler, f, direction, realSamples):
    dimension = f.dimension
    num_iter = samples.shape[1]
    num_samples = samples.shape[0]
    bin_num = 100

    # 1. Histogram of samples
    burnin = int(num_iter * 0.1)
    plt.hist(np.matmul(samples[0,burnin:,:],direction), bins=bin_num, edgecolor='red', density=True,fill=False)
    plt.savefig(f"{sampler}_histogram random direction.png")
    plt.close()
    
    # 2.1 Discrete TV on a random direction 
    number_samples_hist = int(0.2 * num_iter)
    min_number_samples_hist = int(0.1 * num_iter)
    number_hist = 100
    # Not the precise bin_size but it doesn't matter
    bin_size = int((number_samples_hist-min_number_samples_hist) / 100 )
    index_hist = np.linspace(min_number_samples_hist, number_samples_hist, number_hist)
    index_hist = index_hist.astype(int)
    
    projected_sample = np.matmul(realSamples,direction)
    histY, bin_edges_Y = np.histogram(projected_sample, bins=bin_num, density=True)
    distances_random = np.zeros(number_hist)
    projected_sample_random = np.matmul(samples[0,num_iter-number_samples_hist-1:,:],direction)
    for i in range(number_hist):
        distances_random[i] = TV_estimation(projected_sample_random[-index_hist[i]:], histY, bin_edges_Y, bin_num)
    plt.plot(0.5*distances_random)
    plt.title('TV along direction')
    plt.savefig(f"{sampler}_TV_along_direction.png")
    plt.close()
    plt.plot(np.log(distances_random/2))
    plt.title('log TV along direction')
    plt.savefig(f"{sampler}_log_TV_along_direction.png")
    plt.close()
    
    # 2.2 Projection along 'orthogonal'
    direction = direction.ravel()
    random_vector = np.random.rand(f.dimension)
    dot_product = np.dot(direction, random_vector)
    orthogonal = random_vector - dot_product * direction

    projected_sample = np.matmul(realSamples,orthogonal)
    histY, bin_edges_Y = np.histogram(projected_sample, bins=bin_num, density=True)
    distances_random = np.zeros(number_hist)
    projected_sample_random = np.matmul(samples[0,num_iter-number_samples_hist-1:,:],orthogonal)
    for i in range(number_hist):
        distances_random[i] = TV_estimation(projected_sample_random[-index_hist[i]:], histY, bin_edges_Y, bin_num)
    plt.plot(0.5*distances_random)
    plt.title('TV along orthogonal')
    plt.savefig(f"{sampler}_TV_along_orthogonal.png")
    plt.close()
    

    # 3. Errors of the mean measured by L_2 norm /sqrt(dimension). Ensure f.mean is right
    means = np.zeros(num_iter-burnin)
    for i in range(burnin,num_iter):
        means[i-burnin]  = np.linalg.norm(np.mean(samples[:,i-burnin:i,:])- f.mean, ord = 2) / np.sqrt(f.dimension)
    plt.plot(means)
    plt.title('Errors of the mean measured by L_2 norm /sqrt(dimension) ')
    plt.savefig(f"{sampler}_mean_error.png")
    plt.close()

    # 4. Trace Plot for the first coordinate of the first sample
    plt.plot(np.transpose(samples[0,:,0]))
    plt.title('Trace Plot on the 1st coordinate')
    plt.savefig(f"{sampler}_trace_plot.png")
    plt.close()
    
    
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


# Define a function as the original approximate RGO
def generate_samples(step_size, x_y, f):
    dimension = f.dimension
    ite = 0
    while True:
        # samples = np.random.multivariate_normal(mean = x_y, cov = step_size * np.identity(dimension), size = 2)
        samples = f.Gaussian_next(num=2) * np.sqrt(step_size) + x_y
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
def estimate_step_size(step_size, tolerance, y, f, size = 100, reduce = 0.5,  ratio=0.5):
    dimension = f.dimension
    # Compute the desired subexponential parameter
    x_y = f.solve1(y, step_size)
    testFunction = lambda C : np.mean(np.exp(np.abs(Y)**(2/(1+f.alpha))/C))-2
    while True:
        # Generate random samples from a Gaussian distribution: \exp^{-(x-x_y)^2/(2\step_size)}
        # samples = np.random.multivariate_normal(mean = x_y, cov = step_size * np.identity(dimension), size = size*2)
        samples = f.Gaussian_next(num=2*size) * np.sqrt(step_size) + x_y
        Y = np.zeros(size)
        for i in range(size):
            gradient = f.firstOrder(x_y)
            a = f.zeroOrder(samples[i])-np.dot(gradient,samples[i])
            b = f.zeroOrder(samples[i+size])-np.dot(gradient,samples[i+size])
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
        while abs(f_mid) > 1e-1:
            if f_mid > 0:
                left = mid
            else:
                right = mid
            mid = (left+right)/2
            f_mid =  testFunction(mid)
        if mid < 1 / ( np.log(6/tolerance) / np.log(2)  * ratio) : 
            break
        else:
            step_size = step_size * reduce
            x_y = f.solve1(y, step_size) 
    return step_size, x_y

# Define the outer loop of proximal sampler
def proximal_sampler(initial_step_size, num_samples, num_iter, f, fixed, adjusted_size=50, tolerance=1e-3, reduce = 0.7,ratio = 0.5):
    dimension = f.dimension

    samples = np.zeros([num_samples,num_iter,dimension])
    Ysamples = np.zeros([num_samples,dimension])
    rejections = np.zeros([num_samples, num_iter])
    step_sizes = np.zeros([num_samples,num_iter])

    # Initialize the samples for both fixed and adaptive versions
    samples[:,0,:] = np.random.multivariate_normal(mean = np.zeros(dimension), cov = np.identity(dimension), size = num_samples)
    for j in range(num_samples):
        Ysamples[j,:] = np.random.multivariate_normal(mean =  np.zeros(dimension), cov = np.identity(dimension), size = 1)
        if fixed == False:
            step_size, x_y = estimate_step_size(initial_step_size, tolerance, Ysamples[j,:], f, size = 100, reduce = reduce, ratio = ratio)
            step_sizes[j,0] = step_size
    
    if fixed == True:    
        print(f'fixed sampling') 
        for i in range(1,num_iter):
            for j in range(num_samples):
                x_y = f.solve1(Ysamples[j,:], initial_step_size)
                samples[j,i,:],ite = generate_samples(initial_step_size, x_y, f)
                # Ysamples[j,:] = np.random.multivariate_normal(mean = samples[j,i,:], cov = initial_step_size * np.identity(dimension), size = 1)
                Ysamples[j,:] = f.Gaussian_next() * np.sqrt(initial_step_size) + samples[j,i,:]
                rejections[j,i] = ite
            if i % 100 == 0:
                print(f"Steps:{i}")
                if f.times2 > 0:
                    print(f"Averaged optimization steps of the new one: {f.ite2/f.times2}")
                print(f"Averaged rejection steps : {np.mean(rejections[:,i])}")
        
        return samples
    
    if fixed == False:
        print(f'adaptive sampling')
        for i in range(1,num_iter):
            for j in range(num_samples):
                # if i < 100 or (i >= 100 and np.random.uniform(0,1) < 0.001):
                if True:
                    dynamic_size = 100 if i < 5 else adjusted_size
                    if i % 20 == 0:
                        step_size, x_y = estimate_step_size(1/reduce*step_sizes[j,i-1], tolerance, Ysamples[j,:], f, size = dynamic_size, reduce = reduce, ratio=ratio)
                    else:
                        step_size, x_y = estimate_step_size(step_sizes[j,i-1], tolerance, Ysamples[j,:], f, size = dynamic_size, reduce = reduce, ratio=ratio)

                else:
                    x_y = f.solve1(Ysamples[j,:], step_sizes[j,i-1])
                    step_size = step_sizes[j,i-1]
                samples[j,i,:],ite = generate_samples(step_size, x_y, f)
                # Ysamples[j,:] = np.random.multivariate_normal(mean = samples[j,i,:], cov = step_size * np.identity(dimension), size = 1)
                Ysamples[j,:] = f.Gaussian_next() * np.sqrt(step_size) + samples[j,i,:]
                step_sizes[j,i] = step_size
                rejections[j,i] = ite  
            
            # statistics for the first sample
            if i % 100 == 0:
                print(f"Steps:{i}")
                print(f"Averaged_step_size:{np.mean(step_sizes[:,i])}")
                if f.times2 > 0:
                    print(f"Averaged optimization steps of the new one: {f.ite2/f.times2}")
                print(f"Averaged rejection steps : {np.mean(rejections[:,i])}")
    return samples, step_sizes

def MALA(Target, eta, num_iter, numSamples): 
    Xsamples = np.zeros([numSamples, num_iter,Target.dimension])
    Xsamples[:,0,:] = np.random.multivariate_normal(mean = np.zeros(Target.dimension), cov = np.identity(Target.dimension), size = numSamples)
    standard_Gaussians = np.random.multivariate_normal(mean = np.zeros(Target.dimension), cov = np.identity(Target.dimension), size = num_iter * numSamples)
    Us = np.random.uniform(0,1,size=num_iter*numSamples)
    standard_Gaussians = standard_Gaussians.reshape(numSamples,num_iter,Target.dimension)
    Us = Us.reshape(numSamples,num_iter)
    for j in range(numSamples):
        for i in range(num_iter-1):
            # print(f'MALA:{i}') if i % 500 ==0 else None
            standard_Gaussian = standard_Gaussians[j,i,:]
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
            U = Us[j,i]
            if U <= acc_prob:
                x = y
            Xsamples[j,i+1,:] = x
    return Xsamples