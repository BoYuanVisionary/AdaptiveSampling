import numpy as np
import targets 
import matplotlib.pyplot as plt
from scipy.integrate import quad
import scipy.stats
from sklearn.covariance import EmpiricalCovariance

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

def target_func(dimension, if_simple = True):
    
    if if_simple == True:
        t1 = targets.SemiGaussianTarget(dimension = dimension, mean = np.zeros(dimension), inverse = np.identity(dimension), alpha = 1 , prob = 1)
        Target = targets.TargetMixture(t1)

    if if_simple == False:
    
        a = np.random.multivariate_normal(mean = np.zeros(dimension), cov =  np.identity(dimension), size = 1)
        a = (np.transpose(a) / np.linalg.norm(a, ord = 2)) /2
        a = a[0,:]
        
        
        t1 = targets.SemiGaussianTarget(dimension = dimension, mean = a, inverse = np.identity(dimension), alpha = 1 , prob = 1/2)
        t2 = targets.SemiGaussianTarget(dimension = dimension, mean = -a, inverse = np.identity(dimension), alpha = 1 , prob = 1/2)
        Target = targets.TargetMixture(t1,t2)
    
    return Target

def target_funnel():
    
    t = targets.NFarget()
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


# def estimate_W2_Gaussian(samples, trueMean, trueCov):
#     est_mean = np.mean(samples,axis=0)
#     w2s = []
#     iteration = samples.shape[1]
#     for j in range(iteration):
#         current_samples = samples[:,j,:]
#         emMean = est_mean[j,:]
#         emCov = EmpiricalCovariance(assume_centered=False).fit(current_samples).covariance_ 
#         w = np.sqrt(np.linalg.norm(emMean - trueMean, ord=2) ** 2 +
#                    np.trace(emCov + trueCov - 2 * scipy.linalg.sqrtm(scipy.linalg.sqrtm(emCov) @ trueCov @ scipy.linalg.sqrtm(emCov))))
#         w2s.append(w)
#     return w2s

def estimate_W2_Gaussian(samples, trueMean, trueCov):
    # samples shape: (n_samples, n_dimensions, n_features)
    est_mean = np.mean(samples, axis=0)  # shape: (n_dimensions, n_features)
    w2s = []
    n_samples, n_dimensions, n_features = samples.shape

    for j in range(n_dimensions):
        current_samples = samples[:, j, :]  # shape: (n_samples, n_features)
        emMean = est_mean[j, :]  # shape: (n_features,)
        emCov = EmpiricalCovariance(assume_centered=False).fit(current_samples).covariance_  # shape: (n_features, n_features)
        
        # Ensure covariance matrices are positive semi-definite
        emCov = (emCov + emCov.T) / 2
        trueCov = (trueCov + trueCov.T) / 2
        
        # Compute the Wasserstein-2 distance
        sqrt_emCov = scipy.linalg.sqrtm(emCov)
        term = scipy.linalg.sqrtm(sqrt_emCov @ trueCov @ sqrt_emCov)
        w = np.sqrt(np.linalg.norm(emMean - trueMean, ord=2) ** 2 + np.trace(emCov + trueCov - 2 * term))
        w2s.append(w)
        w_imaginary = np.imag(w)
        if np.abs(w_imaginary) > 0.01:
            print(f"warning: imaginary part:{w_imaginary}")

    return w2s