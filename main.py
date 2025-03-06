# Define a function that generates samples approximate RGO. The target is defined in Potential class.
import numpy as np
import random
from utils import *

def fixed_sampler(seed, dimension, numIter, numSamples,Target):  
    random.seed(seed)
    np.random.seed(seed)

    hatC = (1 + Target.alpha) * (1 / Target.alpha)**(Target.alpha / (1 + Target.alpha)) * (1 / np.pi)**(2 / (1 + Target.alpha)) * 2**((-1 - 2 * Target.alpha) / (1 + Target.alpha))
    min_step_size = hatC / (120 * np.log(6 / 0.01) / np.log(2) * Target.L_alpha * np.sqrt(Target.dimension))
    print(f"min_step_size={min_step_size}")

    # Xsamples_fixed_1 = proximal_sampler(initialStep, numSamples, numIter, f=Target, fixed=True)
    # Xsamples_fixed_2 = proximal_sampler(1e-4, numSamples, numIter, f=Target, fixed=True)
    # Xsamples_fixed_3 = proximal_sampler(1e-5, numSamples, numIter, f=Target, fixed=True)
    # Xsamples_adaptive, step_sizes = proximal_sampler(initialStep, numSamples, numIter, f=Target, fixed=False)
    # Xsamples_MALA_1 = MALA(Target, 1e-3, numIter, numSamples)
    # Xsamples_MALA_2 = MALA(Target, 1e-4, numIter, numSamples)
    # Xsamples_MALA_3 = MALA(Target, 1e-5, numIter, numSamples)
    # Xsamples_fixed_1 = proximal_sampler(1e-3, numSamples, numIter, f=Target, fixed=True)
    # Xsamples_fixed_2 = proximal_sampler(1e-2, numSamples, numIter, f=Target, fixed=True)
    # Xsamples_fixed_3 = proximal_sampler(1e-1, numSamples, numIter, f=Target, fixed=True)
    # Xsamples_adaptive, step_sizes = proximal_sampler(initialStep, numSamples, numIter, f=Target, fixed=False)

    # Xsamples_MALA_1 = proximal_sampler(5e-3, numSamples, numIter, f=Target, fixed=True)
    # Xsamples_MALA_2 = proximal_sampler(5e-2, numSamples, numIter, f=Target, fixed=True)
    # Xsamples_MALA_3 = proximal_sampler(5e-1, numSamples, numIter, f=Target, fixed=True)
    
    # samples = {
    #     'PS_10': Xsamples_fixed_1,
    #     'PS_1': Xsamples_fixed_2,
    #     'PS_0.1': Xsamples_fixed_3,
    #     'Adaptive_10': Xsamples_adaptive,
    #     'MALA_10': Xsamples_MALA_1,
    #     'MALA_1': Xsamples_MALA_2,
    #     'MALA_0.1': Xsamples_MALA_3
    # }
    initialStep = 5
    Xsamples_fixed = proximal_sampler(initialStep, numSamples, numIter, f=Target, fixed=True)
    Xsamples_adaptive, step_sizes = proximal_sampler(initialStep, numSamples, numIter, f=Target, fixed=False, ratio = 0.01)
    Xsamples_MALA = MALA(Target, initialStep, numIter, numSamples)
    
    np.save(f'Fixed_{initialStep}_{seed}_{dimension}_{target_name}.npy',Xsamples_fixed)
    np.save(f'step_size_{initialStep}_{seed}_{dimension}_{target_name}.npy',step_sizes)
    np.save(f'Adaptive_{initialStep}_{seed}_{dimension}_{target_name}.npy', Xsamples_adaptive)
    np.save(f'MALA_{initialStep}_{seed}_{dimension}_{target_name}.npy',Xsamples_MALA)
    
    initialStep = 1
    Xsamples_fixed = proximal_sampler(initialStep, numSamples, numIter, f=Target, fixed=True)
    Xsamples_adaptive, step_sizes = proximal_sampler(initialStep, numSamples, numIter, f=Target, fixed=False, ratio = 0.01)
    Xsamples_MALA = MALA(Target, initialStep, numIter, numSamples)
    
    np.save(f'Fixed_{initialStep}_{seed}_{dimension}_{target_name}.npy',Xsamples_fixed)
    np.save(f'step_size_{initialStep}_{seed}_{dimension}_{target_name}.npy',step_sizes)
    np.save(f'Adaptive_{initialStep}_{seed}_{dimension}_{target_name}.npy', Xsamples_adaptive)
    np.save(f'MALA_{initialStep}_{seed}_{dimension}_{target_name}.npy',Xsamples_MALA)
    
    initialStep = 0.2
    Xsamples_fixed = proximal_sampler(initialStep, numSamples, numIter, f=Target, fixed=True)
    Xsamples_adaptive, step_sizes = proximal_sampler(initialStep, numSamples, numIter, f=Target, fixed=False, ratio = 0.01)
    Xsamples_MALA = MALA(Target, initialStep, numIter, numSamples)
    
    np.save(f'Fixed_{initialStep}_{seed}_{dimension}_{target_name}.npy',Xsamples_fixed)
    np.save(f'step_size_{initialStep}_{seed}_{dimension}_{target_name}.npy',step_sizes)
    np.save(f'Adaptive_{initialStep}_{seed}_{dimension}_{target_name}.npy', Xsamples_adaptive)
    np.save(f'MALA_{initialStep}_{seed}_{dimension}_{target_name}.npy',Xsamples_MALA)

    
    
def run_experiments(seeds, dimension, numIter, numSamples, target_name='Gaussian'):

    if target_name=='Gaussian':
        Target = target_Gaussian(dimension)
        direction = np.random.multivariate_normal(mean=np.zeros(dimension), cov=np.identity(dimension), size=1)
        direction = np.transpose(direction) / np.linalg.norm(direction, ord=2)
    elif target_name=='MixedGaussian':
        Target, direction = target_MixedGaussian(dimension)
    elif target_name=='funnel':
        Target = target_funnel(dimension)
        direction = np.zeros([dimension, 1])
        direction[0,0] = 1
    Target.Gaussian(length= numSamples * numIter)

    for seed in seeds:
        fixed_sampler(seed, dimension, numIter, numSamples, Target)



repeat_experiments = 3
seeds = [i+3 for i in range(repeat_experiments)]
numIter = 10000
numSamples = 10
target_name = 'MixedGaussian'
dimensions = [128]
for dimension in dimensions:
    run_experiments(seeds, dimension, numIter, numSamples, target_name)




            

