# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 10:38:17 2019

@author: xli63
"""
import numpy as np
from sklearn.utils import check_random_state
from random import sample
import random
import skimage 

def _dynamic_max_trials(n_inliers, n_samples, min_samples, probability):
    """Determine number trials such that at least one outlier-free subset is
    sampled for the given inlier/outlier ratio.
    Parameters
    ----------
    n_inliers : int
        Number of inliers in the data.
    n_samples : int
        Total number of samples in the data.
    min_samples : int
        Minimum number of samples chosen randomly from original data.
    probability : float
        Probability (confidence) that one outlier-free sample is generated.
    Returns
    -------
    trials : int
        Number of trials.
    """
    if n_inliers == 0:
        return np.inf

    nom = 1 - probability
    if nom == 0:
        return np.inf

    inlier_ratio = n_inliers / float(n_samples)
    denom = 1 - inlier_ratio ** min_samples
    if denom == 0:
        return 1
    elif denom == 1:
        return np.inf

    nom = np.log(nom)
    denom = np.log(denom)
    if denom == 0:
        return 0

    return int(np.ceil(nom / denom))



def random_select_tile (spl_tile_ls,min_samples):
    tile_select_ratio = int( np.ceil( min_samples/ len(spl_tile_ls)) )
    crucial_candidates = np.array([],dtype= np.int)    
    # gurantee each tile have one item to be select (uniform distribution)
    for spl_tile_idxs in spl_tile_ls:   
        random.shuffle(spl_tile_idxs)          
        crucial_candidates = np.concatenate( ( crucial_candidates,
                                             spl_tile_idxs[:tile_select_ratio] ),
                                             axis=0 )# make sure the miss tile have keypoint to select                   
    random.shuffle(crucial_candidates)
    return crucial_candidates[:min_samples]

def ransac_tile(data, model_class, min_samples, residual_threshold,
           is_data_valid=None, is_model_valid=None,
           max_trials=100, stop_sample_num=np.inf, stop_residuals_sum=0,
           stop_probability=1, random_state=None, initial_inliers=None ,spl_tile_ls = None):
    
    """Fit a model to data with the RANSAC (random sample consensus) algorithm.
    RANSAC is an iterative algorithm for the robust estimation of parameters
    from a subset of inliers from the complete data set. Each iteration
    performs the following tasks:
    1. Select `min_samples` random samples from the original data and check
       whether the set of data is valid (see `is_data_valid`).
    2. Estimate a model to the random subset
       (`model_cls.estimate(*data[random_subset]`) and check whether the
       estimated model is valid (see `is_model_valid`).
    3. Classify all data as inliers or outliers by calculating the residuals
       to the estimated model (`model_cls.residuals(*data)`) - all data samples
       with residuals smaller than the `residual_threshold` are considered as
       inliers.
    4. Save estimated model as best model if number of inlier samples is
       maximal. In case the current estimated model has the same number of
       inliers, it is only considered as the best model if it has less sum of
       residuals.
    These steps are performed either a maximum number of times or until one of
    the special stop criteria are met. The final model is estimated using all
    inlier samples of the previously determined best model.
    Parameters
    ----------
    data : [list, tuple of] (N, ...) array
        Data set to which the model is fitted, where N is the number of data
        points and the remaining dimension are depending on model requirements.
        If the model class requires multiple input data arrays (e.g. source and
        destination coordinates of  ``skimage.transform.AffineTransform``),
        they can be optionally passed as tuple or list. Note, that in this case
        the functions ``estimate(*data)``, ``residuals(*data)``,
        ``is_model_valid(model, *random_data)`` and
        ``is_data_valid(*random_data)`` must all take each data array as
        separate arguments.
    model_class : object
        Object with the following object methods:
         * ``success = estimate(*data)``
         * ``residuals(*data)``
        where `success` indicates whether the model estimation succeeded
        (`True` or `None` for success, `False` for failure).
    min_samples : int in range (0, N)
        The minimum number of data points to fit a model to.
    residual_threshold : float larger than 0
        Maximum distance for a data point to be classified as an inlier.
    is_data_valid : function, optional
        This function is called with the randomly selected data before the
        model is fitted to it: `is_data_valid(*random_data)`.
    is_model_valid : function, optional
        This function is called with the estimated model and the randomly
        selected data: `is_model_valid(model, *random_data)`, .
    max_trials : int, optional
        Maximum number of iterations for random sample selection.
    stop_sample_num : int, optional
        Stop iteration if at least this number of inliers are found.
    stop_residuals_sum : float, optional
        Stop iteration if sum of residuals is less than or equal to this
        threshold.
    stop_probability : float in range [0, 1], optional
        RANSAC iteration stops if at least one outlier-free set of the
        training data is sampled with ``probability >= stop_probability``,
        depending on the current best model's inlier ratio and the number
        of trials. This requires to generate at least N samples (trials):
            N >= log(1 - probability) / log(1 - e**m)
        where the probability (confidence) is typically set to a high value
        such as 0.99, and e is the current fraction of inliers w.r.t. the
        total number of samples.
    random_state : int, RandomState instance or None, optional
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    initial_inliers : array-like of bool, shape (N,), optional
        Initial samples selection for model estimation
    Returns
    -------
    model : object
        Best model with largest consensus set.
    inliers : (N, ) array
        Boolean mask of inliers classified as ``True``.
    """

    best_model = None
    best_inlier_num = 0
    best_inlier_residuals_sum = np.inf
    best_inliers = None
    best_inlier_tile_mean =0

    random_state = check_random_state(random_state)

    # in case data is not pair of input and output, male it like it
    if not isinstance(data, (tuple, list)):
        data = (data, )
    num_samples = len(data[0])

    if not (0 < min_samples < num_samples):
        raise ValueError("`min_samples` must be in range (0, <number-of-samples>)")

    if residual_threshold < 0:
        raise ValueError("`residual_threshold` must be greater than zero")

    if max_trials < 0:
        raise ValueError("`max_trials` must be greater than zero")

    if not (0 <= stop_probability <= 1):
        raise ValueError("`stop_probability` must be in range [0, 1]")

    if initial_inliers is not None and len(initial_inliers) != num_samples:
        raise ValueError("RANSAC received a vector of initial inliers (length %i)"
                         " that didn't match the number of samples (%i)."
                         " The vector of initial inliers should have the same length"
                         " as the number of samples and contain only True (this sample"
                         " is an initial inlier) and False (this one isn't) values."
                         % (len(initial_inliers), num_samples))
                         
    
    # for the first run use initial guess of inliers
    spl_idxs = (initial_inliers if initial_inliers is not None
                    else random_select_tile (spl_tile_ls, min_samples) )
#                else random_state.choice(num_samples, min_samples, replace=False))    
#    import pdb ; pdb.set_trace()

    
#    spl_idxs_crucial = []
    for num_trials in range(max_trials):
        # do sample selection according data pairs
        samples = [d[spl_idxs] for d in data]
        # for next iteration choose random sample set and be sure that no samples repeat
#        spl_idxs = random_state.choice(num_samples, min_samples, replace=False)
#        spl_idxs= np.concatenate ((np.array( spl_idxs_crucial,dtype=np.int),
#                                   spl_idxs[:min_samples-len(spl_idxs_crucial)]),axis=0 )
        
        spl_idxs = random_select_tile (spl_tile_ls, min_samples) 
        # optional check if random sample set is valid
        if is_data_valid is not None and not is_data_valid(*samples):
            continue

        # estimate model for current random sample set
        sample_model = model_class()

        success = sample_model.estimate(*samples)
        # backwards compatibility
        if success is not None and not success:
            continue

        # optional check if estimated model is valid
        if is_model_valid is not None and not is_model_valid(sample_model, *samples):
            continue

        sample_model_residuals = np.abs(sample_model.residuals(*data))
        # consensus set / inliers
        
        import matplotlib.pyplot as plt
        plt.figure()
        plt.hist(sample_model_residuals,100)
        
        sample_model_inliers = sample_model_residuals < residual_threshold
        
        sample_model_residuals_sum = np.sum(np.abs(sample_model_residuals))

        # choose as new best model if number of inliers is maximal
        sample_inlier_num = np.sum(sample_model_inliers)        
        sample_inlier_tile_perc = [sample_model_inliers[a].sum() / len(sample_model_inliers[a])  
                                        for a in spl_tile_ls]
#        import pdb ; pdb.set_trace()
#        sample_inlier_tile_mean = ( (np.array(sample_inlier_tile_perc)!=0).sum() ) / len(sample_inlier_tile_perc)   # non zero tiles
        sample_inlier_tile_mean = np.mean(sample_inlier_tile_perc)   # non zero tiles

#        sample_inlier_tile_std = np.std(sample_inlier_tile_perc)
#        print ("sample_inlier_tile_mean =", sample_inlier_tile_mean)       
#        print ("sample_inlier_tile_std =", sample_inlier_tile_std)       

        if (
            # more inliers
            sample_inlier_num > best_inlier_num
#             sample_inlier_tile_mean > best_inlier_tile_mean
#             same number of inliers but less "error" in terms of residuals
            or (
                    sample_inlier_num == best_inlier_num
                and 
                sample_model_residuals_sum < best_inlier_residuals_sum
                )
        ):
            best_model = sample_model
            best_inlier_num = sample_inlier_num
            best_inlier_tile_mean = sample_inlier_tile_mean 
            best_inlier_residuals_sum = sample_model_residuals_sum
            best_inliers = sample_model_inliers
            dynamic_max_trials = _dynamic_max_trials(best_inlier_num,
                                                     num_samples,
                                                     min_samples,
                                                     stop_probability)
            if (best_inlier_num >= stop_sample_num
                or best_inlier_residuals_sum <= stop_residuals_sum
                or num_trials >= dynamic_max_trials):
                break
            
#            if sample_inlier_tile_mean == best_inlier_tile_mean: 
#            if sample_inlier_num == best_inlier_num: 
            print("-iter: ", num_trials, "spl_idxs=", len(spl_idxs),
              "sample_inlier_num=", '%.2f'%( best_inlier_num/len(best_inliers)))

            ### local optimal selection  increase the threshold for K times
            thre_multiplier = 10
            iters = 100
            local_residual_threshold = residual_threshold * thre_multiplier
            det_thres = ( local_residual_threshold - residual_threshold ) / ( iters -1)

            local_sample_model_residuals = np.abs(best_model.residuals(*data))
            # consensus set / inliers
            base_inliers = local_sample_model_residuals < local_residual_threshold     
            data_base_inliers = [d[base_inliers] for d in data]
#            base_model = model_class()    
#            base_model.estimate(*data_base_inliers)

            local_min_samples = int( min (base_inliers.sum()/2,10 ) )  
            thres_iter = local_residual_threshold                              # initalize the threshold 

            for r in range (iters): # inner loop)            
                if len (data_base_inliers) > local_min_samples : 
                    local_spl_idxs = random_state.choice(len(data_base_inliers),   
                                                     local_min_samples, replace=False)     
                    local_sample_data = [d[local_spl_idxs] for d in data_base_inliers] 
                    local_sample_model = model_class()    
                    local_sample_model.estimate(*local_sample_data)             # estimate model from samples
                    
                    iter_residuals = np.abs(local_sample_model.residuals(*data))
                    inlier_iter = iter_residuals < thres_iter   
                    temp_inlier_num = np.sum(inlier_iter)  
                    
                    if temp_inlier_num > best_inlier_num:
                        best_inlier_num = temp_inlier_num
                        best_model = local_sample_model
                        best_inliers = inlier_iter
                        print ("--- local updates to ",'%.2f'%(best_inlier_num/len(best_inliers)))
                    thres_iter = thres_iter - det_thres                        

                    
    # estimate final model using all inliers
    if best_inliers is not None:
        # select inliers for each data array
        data_inliers = [d[best_inliers] for d in data]
        best_model.estimate(*data_inliers)    

    return best_model, best_inliers