# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 10:38:17 2019

@author: xli63
"""
import numpy as np
from sklearn.utils import check_random_state
from random import sample
import random
import vis_fcts as vis_fcts
from skimage.transform import warp

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


def hierarchy_shuffle(intervel_min = 10):
    ''' Gurantee the selected ele's width is not less than 10 '''

def random_select_tile (spl_tile_ls,min_samples, crucial_id_tile_ls = [] ):
    '''
    Our improvement of ransac transformation estimation to texture image:
    hieracarchical selection:
    
    2) select min_samples groups
    3) select the element in each group
    '''
    # 
    spl_tile_ls = [i for i in spl_tile_ls if  len(i)>0 ]
    random.shuffle(spl_tile_ls)   
    # tile_group_size = int( np.ceil( len(spl_tile_ls)/min_samples) )    # 
    selected_candidates = []
    # if len(crucial_id_tile_ls ) ==0 :
    # ## First levelm select random tile
    # random_tile_id = spl_tile_ls[]
    # groups = np.split ( np.arange(len(spl_tile_ls)), tile_group_size)     # 1) arrange the big group of tiles, contrain ids

    if min_samples <= len(spl_tile_ls):                             # selecte each one element in each tile
        for group_i  in range ( min_samples ):                      # select first 4 tile ls in shuffled(spl_tile_ls)
            selected_tile = spl_tile_ls[group_i]
            selected_candidate = np.random.choice(selected_tile)
            selected_candidates.append( selected_candidate )
    else:                                                           # min_samples > len(spl_tile_ls)
        max_ids_tile = int( np.ceil( min_samples/len(spl_tile_ls)) )
        remain_cand_nums = min_samples
        # import pdb;pdb.set_trace()
        while remain_cand_nums > 0 :                    
            for spl_tile in  spl_tile_ls:
                random.shuffle(spl_tile)
                selected_candidates += list( spl_tile[:max_ids_tile])
                remain_cand_nums -= len(selected_candidates)
         
    return selected_candidates


def ransac_tile(data, model_class, min_samples =3, residual_threshold =5,
           is_data_valid=None, is_model_valid=None,
           max_trials=100, stop_sample_num=np.inf, stop_residuals_sum=0,
           stop_probability=1, random_state=None, initial_inliers=None ,spl_tile_ls = None,verbose= False, obj_type = "inlier_num",
           source_target= None):
    
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
    
    obj_type:   [ "inlier_num","inlier_tile", "residuals_sum", "fast_diff"]

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
    best_fast_diff = np.inf    # estimated_error_min = 200

    random_state = check_random_state(random_state)

    assert obj_type in [ "inlier_num","inlier_tile", "residuals_sum", "fast_diff"]
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
    # import pdb;pdb.set_trace()

    spl_idxs = (initial_inliers if initial_inliers is not None
                    else random_select_tile (spl_tile_ls, min_samples) )
  
    crucial_id_tile_ls = []
    for num_trials in range(max_trials):
        # do sample selection according data pairs
        # if verbose is True:
        #     print("num_trials=",num_trials) 
        samples = [d[spl_idxs] for d in data]

        # for next iteration choose random sample set and be sure that no samples repeat
        spl_idxs = random_select_tile (spl_tile_ls, min_samples, crucial_id_tile_ls = crucial_id_tile_ls) 
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
        

        # choose as new best model if number of inliers is maximal
        # define the critera
        

        if obj_type != "fast_diff":
            sample_model_residuals = np.abs(sample_model.residuals(*data))
            # consensus set / inliers       
            sample_model_inliers = sample_model_residuals < residual_threshold      

            if obj_type =="inlier_num":
                sample_inlier_num = np.sum(sample_model_inliers)       
                critera = ( sample_inlier_num > best_inlier_num )

            elif obj_type =="inlier_tile":
                sample_inlier_tile_perc = [sample_model_inliers[a].sum() / len(sample_model_inliers[a])  
                                                for a in spl_tile_ls]                  # inlier list over all tiles
                sample_inlier_tile_mean = np.mean(sample_inlier_tile_perc)   
                critera = (sample_inlier_tile_mean > best_inlier_tile_mean)

            elif obj_type =="residuals_sum":
                sample_model_residuals_sum = np.sum(np.abs(sample_model_residuals))

                critera = (sample_model_residuals_sum < best_inlier_residuals_sum)
        else:
            # import pdb;pdb.set_trace()
            __, __,__,estimated_error,__ = vis_fcts.eval_draw_diff ( source_target[1], 
                                                              warp( source_target[0], 
                                                                    inverse_map = sample_model.inverse, 
                                                                    output_shape = source_target[1].shape, cval=0) )    

            critera = ( estimated_error < best_fast_diff)
            # import pdb;pdb.set_trace()

        # update best
        update_indicator = False
        if ( critera and ( np.all(np.isnan(sample_model.params)) == False) ):
            best_model = sample_model
            update_indicator = True
            if obj_type != "fast_diff":
                best_inliers = sample_model_inliers
                if obj_type =="inlier_num":
                    best_inlier_num = sample_inlier_num
                    dynamic_max_trials = _dynamic_max_trials(best_inlier_num,
                                                     num_samples,
                                                     min_samples,
                                                     stop_probability)
                    if (best_inlier_num >= stop_sample_num) or (num_trials >= dynamic_max_trials):
                        break
                    
                    if verbose is True:
                        print("-iter: ", num_trials, "spl_idxs=", spl_idxs,
                            "sample_inlier_num=", '%.2f'%( best_inlier_num/len(best_inliers)))

                elif obj_type =="inlier_tile":
                    best_inlier_tile_mean = sample_inlier_tile_mean 

                elif obj_type =="residuals_sum":
                    best_inlier_residuals_sum = sample_model_residuals_sum
                    if best_inlier_residuals_sum <= stop_residuals_sum:
                        break

            else:
                best_fast_diff = estimated_error
                # import pdb;pdb.set_trace()

                    
    # estimate final model using all inliers
    if update_indicator == True :
        # calculate the inliner 
        sample_model_residuals = np.abs(best_model.residuals(*data))
        # consensus set / inliers       
        best_inliers = sample_model_residuals < residual_threshold      
        
        best_inliers_subset = np.zeros_like(best_inliers)
        # import pdb;pdb.set_trace()
        spl_idxs = random_select_tile (spl_tile_ls, min(best_inliers.sum(), min_samples*1000) ) 
        best_inliers_subset[spl_idxs] = True

        # select a random subset of inliers for each data array
        best_inliers_subset = best_inliers_subset* best_inliers
        data_inliers = [d[best_inliers_subset] for d in data]
        if verbose == True:
            print ("Estimate the Best model...")
        del sample_model,data,success                 # in case num(best_inliers) too large,  model.estimate run out of memory

        best_model.estimate(*data_inliers)    
        if verbose == True:
            print ("Best Model generated")   

    return best_model, best_inliers




