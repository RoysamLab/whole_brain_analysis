

'''
result_dir="[retest]-autoseg-mrcnn1-bb-split-mrcnn2-bb"
python eval_gt.py -c iou_ann \
--pd /project/ece/roysam/xiaoyang/exps/Data/50_plex/jj_final/seg_results/"$result_dir"/pred_atlas \
--gt /project/ece/roysam/xiaoyang/exps/Data/50_plex/jj_final/atlas/ground_truth \
-o /project/ece/roysam/xiaoyang/exps/Data/50_plex/jj_final/seg_results/"$result_dir"/eval

--submit /project/ece/roysam/xiaoyang/exps/Data/50_plex/jj_final/seg_results/"$result_dir"/merged_submission.csv \

'''


import sys, os
import csv
import math
import numpy as np
import skimage
from skimage import exposure, segmentation,morphology,io,img_as_ubyte,measure
from skimage.external import tifffile as tiff
import multiprocessing
from sklearn import mixture
from functools import partial
from itertools import repeat
import h5py
import pandas as pd
import glob
import scipy
import random
from sklearn.metrics import precision_recall_curve, roc_curve
import pickle
# from tqdm import tqdm
import matplotlib        
matplotlib.use('Agg')  # Agg backend runs without a display
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
# Import Mask RCNN
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from tqdm import tqdm
sys.path.insert(0, "/project/ece/roysam/xiaoyang/exps/SegmentationPipeline/mrcnn_Seg/supplement")
import datasets_utils as dt_utils

def drawNCal_F1_IOU_curve(tps,fps,fns,iou_thresholds,fig_fname):
    precision = tps / (tps + fps)
    recall = tps / (tps + fns)
    F1 = 2 * (precision * recall) / (precision + recall)

    BF50 = F1 [ np.where(iou_thresholds==0.5)[0][0]  ]     #BF score/ F1 score
    BF75 = F1 [ np.where(iou_thresholds==0.75)[0][0] ] 

    plt.figure(figsize=(3,2),dpi=200)
    plt.plot(iou_thresholds, F1, "-.")
    plt.xlabel("IoU")
    plt.ylabel("F1-score")
    plt.title("F1score-IoU curve" +"\n" +
            "BF50 = {0:.3f},".format(BF50)+"BF75 = {0:.3f}".format(BF75))
    plt.savefig(fig_fname, bbox_inches="tight", dpi=300)
    plt.close()

    return F1
def label2mask_vis(label):
    label,__,__ = segmentation.relabel_sequential(label)     # relabel start with 1
    label = label.astype(np.float)
    mask = ( label/label.max() * 255) .astype(np.uint8)    # 128,255
    
    # import pdb; pdb.set_trace()

    return mask

def image_eval(  imgid , pd_path, gt_path, iou_thresholds, submit_df=None, error_obj_path = None):
    submit_df = None
    pd_mask_img = io.imread (os.path.join (pd_path,imgid+"_mask.png"))[:,:,0]
    gt_mask_img = io.imread (os.path.join (gt_path,imgid+"_mask.png"))[:,:,0]

    # import pdb; pdb.set_trace()

    if pd_mask_img.max() > 0:        
        pd_bbox, pd_class_id, pd_mask, pd_label = dt_utils.mask2props(pd_mask_img, RemoveEdge = True)  
        gt_bbox, gt_class_id, gt_mask, gt_label = dt_utils.mask2props(gt_mask_img, RemoveEdge = True)                    

        color =  skimage.color.label2rgb( pd_label,bg_label=0)
        # import pdb; pdb.set_trace()
        io.imsave(  os.path.join (pd_path,imgid+"_color.png"),skimage.img_as_ubyte(color))

        pd_score = (np.zeros_like(pd_class_id) +1).tolist()  # default: assigning all the scores to 1
        ## Step 1: calucate all IOUs
        if  submit_df is not None:   # Find gt_match with score id 
            crop_size = [512,512]
            pd_score = []
            max_overlap = 0
            pd_class_id = []
            submit_crop =  submit_df[submit_df['ImageId'] == imgid]
            submit_crop_mask = {}                     
            # find the mask_i closest to the predict mask
            for mask_i in range(pd_mask.shape[2]):
                pd_m = pd_mask[:, :, mask_i]
                for obj_i in submit_crop.index:
                    rle = submit_crop.loc[obj_i]["EncodedPixels"]
                    submit_mask = dt_utils.rle_decode(rle, crop_size)
                    overlap = (submit_mask*pd_m).sum()
                    if overlap >= max_overlap:
                        max_overlap = overlap
                        score = submit_crop.loc[obj_i]["Score"]
                        class_id = submit_crop.loc[obj_i]["Class"]
                
                pd_score.append( score)
                pd_class_id.append( class_id)                                

        #     gt_match, pd_match,overlaps = utils.compute_matches(gt_bbox, np.array(gt_class_id), gt_mask,
        #                 pd_bbox, np.array(pd_class_id), np.array(pd_score),pd_mask,
        #                 iou_threshold = 0)                        
        # else:  
        gt_match, pd_match,overlaps  = dt_utils.compute_matches_bymasks(pd_mask,gt_mask)  # *_match: id for matches, -1 for lost
        Miss,Spurious,Split,Merge,ids = compareMetrics_labels(gt_label, pd_label)

        # import pdb; pdb.set_trace()
        if error_obj_path is not None:   # the saving path for 4 types of error
            orig_img = io.imread(os.path.join (gt_path,imgid+".tif"))    # jpeg for gray, .tiff for 8 channel 
            # 8channel to 7 channels: s: DAPI, Histones, NeuN, S100, Olig 2, Iba1 and RECA
            gray_img_ls = np.split (orig_img,8,axis = 2)
            del gray_img_ls[3] 
            gray_img = np.dstack(gray_img_ls)

            for error_id_key in ids.keys():
                error_type = error_id_key.split("_")[0]
                save_dir = os.path.join(error_obj_path,error_type)
                os.makedirs( save_dir,exist_ok= True)
                if "gt" in error_id_key:
                    for obj in measure.regionprops(gt_label):
                        if obj.label in ids[error_id_key]:                                     # id have erros
                            img_id = imgid  + "_" + str(obj.label)
                            minr, minc, maxr, maxc = obj.bbox                        
                            gt_mask = (obj.image*255).astype(np.uint8)                          # missing gt,
                            pd_mask = label2mask_vis(pd_label[minr:maxr,minc:maxc])

                            io.imsave(os.path.join(save_dir,img_id+"-img.tif"), gray_img[minr:maxr,minc:maxc])     # save img                                    
                            io.imsave(os.path.join(save_dir,img_id+"-gt.png"), gt_mask )           # save gt_mask
                            io.imsave(os.path.join(save_dir,img_id+"-pd.png"), pd_mask )             # save pd_mask
                elif "pd" in error_id_key:
                    for obj in measure.regionprops(pd_label):
                        if obj.label in ids[error_id_key]:
                            img_id = imgid  + "_" + str(obj.label)
                            minr, minc, maxr, maxc = obj.bbox                        
                            gt_mask = label2mask_vis(gt_label[minr:maxr,minc:maxc] )
                            pd_mask = (obj.image*255).astype(np.uint8)                              #single obj

                            io.imsave(os.path.join(save_dir,img_id+"-img.tif"), gray_img[minr:maxr,minc:maxc])     # save img                                    
                            io.imsave(os.path.join(save_dir,img_id+"-gt.png"), gt_mask )           # save gt_mask
                            io.imsave(os.path.join(save_dir,img_id+"-pd.png"), pd_mask )             # save pd_mask
            # import pdb; pdb.set_trace()

        if submit_df is not None:
            AP50, __, __, __ =\
                utils.compute_ap(gt_bbox, np.array(gt_class_id), gt_mask,
                        pd_bbox, np.array(pd_class_id), np.array(pd_score),pd_mask,
                        iou_threshold = 0.5)
            AP75, __, __, __ =\
                utils.compute_ap(gt_bbox, np.array(gt_class_id), gt_mask,
                        pd_bbox, np.array(pd_class_id), np.array(pd_score),pd_mask,
                        iou_threshold = 0.75)                                        
            # 
            mAP = utils.compute_ap_range(gt_bbox, np.array(gt_class_id), gt_mask,
                        pd_bbox, np.array(pd_class_id), np.array(pd_score),pd_mask,
                            )  #0.5 to 0.95 with increments of 0.05                                       

        # calculate TP
        match_ious = overlaps[np.arange(len(overlaps))[pd_match > -1], 
                        pd_match[pd_match > -1].astype(np.int)]   # in this way only the iou good are selected
        match_array = match_ious[None, :] >= iou_thresholds[:, None]
        num_matchs = np.sum(match_array, axis=-1)

        rs = {}
        rs["tp"] = num_matchs
        rs["fp"] = len(pd_match)- num_matchs
        rs["fn"] = len(gt_match) - num_matchs       
        rs["iou_pd"] = match_ious
        miss_ious = np.max(overlaps[:, np.arange(len(gt_match))[gt_match == -1]], axis=0)    # assign not matched to the max values
        rs["iou_gt"] = np.concatenate([match_ious, miss_ious])

        rs["metrics"] = { "mIoU_gt": rs["iou_gt"].mean(),
                            "mIoU_pd": rs["iou_pd"].mean(),
                            "Miss" : Miss/len(gt_match) ,
                            "Spurious": Spurious/len(gt_match) ,
                            "Split": Split/len(gt_match),
                            "Merge": Merge/len(gt_match)
                        }
        
        __ = drawNCal_F1_IOU_curve(rs["tp"],rs["fp"],rs["fn"],iou_thresholds,fig_fname = os.path.join (pd_path,imgid+"_F1-Iou_Curve.png"))

    else:
        
        return None 

    return rs 



def iou_ann(pd_path,gt_path,output_dir,mp=True, 
            whole_img_path=None,submit_df=None,error_obj_path= None):

    # '''
    # Author: Pengyu Yuan , Multiprocesing by Rebecca
    # pd_path: the folder contain the pdicted images:  "*_mask.png"
    # gt_path: the folder contain the groud Truth images:  "*_mask.png"
    # '''

    if  submit_df is not None:
        submit_df = pd.read_csv(submit_df)
    os.makedirs( output_dir,exist_ok= True)

    # iou_thresholds = np.arange(0.5, 1.0, 0.05) if iou_thresholds is None else iou_thresholds  =
    iou_thresholds = np.arange(0, 1.05, 0.05) # if iou_thresholds is None else iou_thresholds
    # num_iou_ids = 21   #     # num_iou_ids the number of intervals of the IoUs
    # iou_thresholds = np.linspace(0, 1, num_iou_ids)     # 0...1 with interval 21, [0,0.05,0.01... ,0.95]
    tplist, fplist, fnlist = [], [], []
    ious_list_pd , ious_list_gt,clumps_ls= []   ,[],[]    
    # Search for image names
    imgid_ls ,mIoU_ls = [],[]
    for image_mask in os.listdir(pd_path):
        if "_mask.png" in image_mask :                
            if image_mask in os.listdir(gt_path) :# and it < 2:                    
                imgid = image_mask.split("_mask.png")[0]
                imgid_ls.append(imgid)
    # process per iamge

    if mp:
        try:
            cpus = multiprocessing.cpu_count()
        except NotImplementedError:
            cpus = 2  # arbitrary default
        with multiprocessing.Pool(processes=cpus) as pool:
            r_ls = pool.starmap(image_eval,  zip( imgid_ls, repeat(pd_path), repeat(gt_path) ,
                            repeat(iou_thresholds)  ,repeat(submit_df) ,repeat(error_obj_path)     ))
        #(  imgid , pd_path, gt_path, iou_thresholds, submit_df=None, error_obj_path = None):
    else:    

        r_ls = []
        for image_mask in imgid_ls:
            r_ls.append(image_eval(image_mask, pd_path,gt_path,iou_thresholds,submit_df,error_obj_path))


    metrics_dic = {}            # metrics for images
    valid_imgid_ls = []
    for rs , imgid in zip( r_ls, imgid_ls):
        if rs is not None:
            valid_imgid_ls.append(imgid)

            tplist.append(rs["tp"])
            fplist.append(rs["fp"])
            fnlist.append(rs["fn"])
            ious_list_pd.append(rs["iou_pd"])
            ious_list_gt.append(rs["iou_gt"])

            metrics = rs["metrics"]
            for key in rs["metrics"]:
                 # initalize the dict key and the list
                if key not in metrics_dic.keys() :
                    metrics_dic[key] = []   
                metrics_dic[key].append(rs["metrics"][key])

    '''    ### save the metric in image level'''    
    atlas_dict = {}
    atlas_dict["imgid"] = valid_imgid_ls
    for key in rs["metrics"]:
        atlas_dict[key] = metrics_dic[key]
    # import pdb; pdb.set_trace()

    atlas_df = pd.DataFrame.from_dict(atlas_dict)        
    atlas_df.to_csv(os.path.join(output_dir,"atlas_eval.csv"))

    '''    ### save the evaluations overall'''    
    ious_list_pd_all = np.concatenate(ious_list_pd)
    ious_list_gt_all = np.concatenate(ious_list_gt)

    mIoU_pd = ious_list_pd_all.mean()                  # all detected iou/ total number of ojects
    mIoU_gt = ious_list_gt_all.mean()                  # all detected iou/ total number of ojects

    tplist = np.array(tplist)
    fplist = np.array(fplist)
    fnlist = np.array(fnlist)

    tps = np.sum(tplist, axis=0)
    fps = np.sum(fplist, axis=0)
    fns = np.sum(fnlist, axis=0)

    F1 = drawNCal_F1_IOU_curve(tps,fps,fns,iou_thresholds,fig_fname = os.path.join(output_dir, "F1score-IoU_final.png"))

    with open(os.path.join(output_dir, "results_final.pkl"), "wb") as f:       # write result in to file
        pickle.dump([tplist, fplist, fnlist, ious_list_pd_all, F1, iou_thresholds,ious_list_gt_all], f)
    
    plt.hist(ious_list_pd_all, 50, density=True)
    plt.xlabel("IoU")
    plt.ylabel("Density")
    plt.title("Histogram of IoU " +"\n" +
              "mIoU_pd = {0:.3f}".format(mIoU_pd) + 
              ",mIoU_gt = {0:.3f}".format(mIoU_gt)    
              )    
    plt.grid()
    plt.savefig(os.path.join(output_dir, "histIoU_final.png"), bbox_inches="tight", dpi=300)
    plt.close()

    print ( "mIoU_pd = {0:.3f}  ".format(mIoU_pd)  +
            "mIoU_gt = {0:.3f}\n".format(mIoU_gt)  )
            # "mAP50 = {0:.3f},".format(mAP)+
            # "BF50 = {0:.3f},".format(BF50)+
            # "BF75 = {0:.3f} ".format(BF75))


    return atlas_df

def compareMetrics_labels(gt_labels, pd_labels):
    '''
    Return the number of   Miss,    Spurious(False positive ),  Split,  Merge
    '''
    def missing (gt_labels,pd_labels):
        miss_labels = gt_labels.copy()
        miss_labels[pd_labels>0] = 0       # only extract the gt labels never been labeled 
        miss_ids = np.unique(miss_labels) 

        miss_numbers = 0
        ids = []                                # the missing id of to gt_labels
        for obj_miss_id in np.unique( miss_ids):
            if obj_miss_id > 0:
                perc = ( miss_labels == obj_miss_id).sum() / (gt_labels == obj_miss_id).sum()
                if perc > 0.7:
                    miss_numbers +=1 
                    ids.append(obj_miss_id)
        return miss_numbers,ids

    def splitting (gt_labels, pd_labels):
        split_numbers = 0
        ids =[]
        for obj in measure.regionprops(gt_labels):
            pd_label_crop = pd_labels[ obj.coords[:,0], obj.coords[:,1]]    # all the pixels to label
            if pd_label_crop.size > 0:
                perc_ls = []                                                #
                for obj_pd_id in np.unique( pd_label_crop):
                    if obj_pd_id > 0:
                        perc_ls.append (  (pd_label_crop==obj_pd_id).sum() / pd_label_crop.sum())
                if len(perc_ls) > 0:
                    if min(perc_ls) > 0.25:                                  # the splited cell should be at lear 20%
                        split_numbers +=1 
                        ids .append (obj.label)
        return split_numbers,ids

    ids ={}
    Miss    , ids["miss_gt"]     = missing (gt_labels,pd_labels)
    Spurious, ids["spurious_pd"] = missing (pd_labels,gt_labels)
    Split   , ids["split_gt"]    = splitting (gt_labels, pd_labels)
    Merge   , ids["merge_pd"]    = splitting (pd_labels, gt_labels)

    # True positive predicted cells
    TF_pd_ls = []
    for obj in measure.regionprops(gt_labels):
        if obj.label not in ids["spurious_pd"]  and obj.label not in ids["merge_pd"] :
            TF_pd_ls.append(obj.label)
    ids["TF_pd"]    = TF_pd_ls 

    return Miss,Spurious,Split,Merge,ids

def iou_labels_calculation(gt_labels, pd_labels):
    # the number of detected object don't have to be the same 
    # the size of 2 label images have to the same 
    extend_margin = 50
    iou_ls = []
    label_ls = []             # {"1":0.3,"2":0.88}    
    pdlabel_ls =[]
    for obj in measure.regionprops(gt_labels):
        gt_label_id  = obj.label 
        gt_coords    = obj.coords                                                                               # (N, 2) ndarray Coordinate list (row, col) of the region.
        (xmin, ymin, xmax, ymax)  = obj.bbox               
        suspicious_shift_height = extend_margin  if xmin> extend_margin  else 0
        suspicious_shift_width  = extend_margin  if ymin> extend_margin  else 0
        suspicious_gt_labels = gt_labels[ xmin-suspicious_shift_height : xmax + extend_margin, 
                                          ymin-suspicious_shift_width  : ymax + extend_margin  ]            # extract the labels of subpicious window from previous merged results
        suspicious_pd_labels = pd_labels[ xmin-suspicious_shift_height : xmax + extend_margin, 
                                          ymin-suspicious_shift_width  : ymax + extend_margin  ]         # extract the labels of subpicious window from previous merged results
        
        if len(pd_labels[gt_coords[:,0],gt_coords[:,1]]) == 0:
            iou = 0
        else:
            pd_label_ids_candidates = pd_labels[gt_coords[:,0],gt_coords[:,1]]                              # read all the pixel labels 
            pd_label_ids_candidates = pd_label_ids_candidates[pd_label_ids_candidates>0]
            pd_label_id = 0
            if len(pd_label_ids_candidates)> 0:
                df_pd = pd.DataFrame(pd_label_ids_candidates)
                pd_label_id =  df_pd.mode()[0].values[0]                                                    # the corresponding highest frequent id in pdict labels 
        
                masks1_coords = np.where(suspicious_gt_labels == gt_label_id)
                masks2_coords = np.where(suspicious_pd_labels == pd_label_id)
                
                i,u = dt_utils.compute_overlaps_masks_coords(masks1_coords, masks2_coords)
                iou = i / u
            else:
                iou = 0
        iou_ls.append(iou)        
        label_ls.append(gt_label_id)
        pdlabel_ls.append(pd_label_id)

    iou_ls = np.array(iou_ls)
    mIoU = iou_ls.mean()
    good_iou_rate      = sum(iou_ls>0.5)/len(iou_ls)    
    excellent_iou_rate = sum(iou_ls>0.8)/len(iou_ls)
    
    iou_df = pd.DataFrame(data = {  "obj_id": label_ls,
                                    "iou_ls": iou_ls ,
                                    'cp_id' : pdlabel_ls,
                                    })
    
    return mIoU,good_iou_rate,excellent_iou_rate,iou_df

def draw_ioufmap(iou_df,output_dir):
    label_mask = np.load("/project/ece/roysam/xiaoyang/exps/Data/50_plex/jj_final/atlas/atlabs_labelimg.npy")
    label_names_df = pd.read_csv("/project/ece/roysam/xiaoyang/exps/Data/50_plex/jj_final/atlas/label_names.csv")
    label_names_df= label_names_df.set_index("crop_img")

    iou_fmap = np.zeros_like(label_mask,dtype=np.float)

    # import pdb; pdb.set_trace()
    for m_i in iou_df.index:
        
        imgid = iou_df["imgid"][m_i].split("_mask.png")[0]
        iou = iou_df["mIoU_gt"][m_i]   #mIoU
        if imgid in list(label_names_df.index):
            label_index =  label_names_df["label"][imgid]        
            iou_fmap[label_mask == label_index]= iou
    iou_fmap[label_mask == 0]= np.nan
    plt.figure()
    # plt.imshow(iou_fmap,vmin=iou.min(),vmax=iou.max(),cmap="hot")  #
    plt.imshow(iou_fmap,vmin=0.5,vmax=0.7,cmap="hot")  #

    plt.colorbar()
    plt.savefig(os.path.join(output_dir,"iou_atlas_maps.png"),dpi=300)


if __name__ == '__main__':
    import argparse,time
    import matplotlib
#     # Agg backend runs without a display
    matplotlib.use('Agg')
    tic = time.time()   
    # Parse command line arguments

    parser = argparse.ArgumentParser(
        description='Calculate IOU over 2 whole brain label mask')

    parser.add_argument("--command","-c",
                        metavar="<command['iou','ann']>", default = 'iou',
                        help="'iou', 'iou_ann' ,'iou_map'")
    parser.add_argument('--output_dir',"-o", required= False,
                        metavar = "/path/to/maskfile/",
                        default = 'iou_df.csv',
                        help='Full name to save as result ')
    parser.add_argument('--pd', required=False,
                        metavar="/path/to/dataset/",
                        default = "/uhpc/roysam/xiaoyang/exps/NuclearSeg_DNN/Mask_RCNN/datasets/CHN50/[DAPI+Histones]208835seeds_Labels.out",
                        help='pd_labels txt')   
    parser.add_argument('--gt', required=False,
                        metavar="/path/to/dataset/",
                        default = "/project/ece/roysam/xiaoyang/exps/Data/50_plex/jj_final/atlas/ground_truth",
                        help='gt_labels txt')
    parser.add_argument('-a','--imadjust', required=False,
                        default = '1',type = str, 
                        help='whether to adjust the image')                                                            
    parser.add_argument('--mp', required=False, type=str,
                        default = "True",
                        help='multiprocessing or not')   
    parser.add_argument('--error_obj_path', required=False, type=str,
                        default = None,
                        help='The path to output the 4 types of error: Miss, surpious,split,merge')   
    args, _ = parser.parse_known_args() 

    if args.command == "iou_ann" or args.command == "iou_map" :
        parser.add_argument('--submit', required=False,
                            metavar="/path/to/dataset/",
                            default = None,
                            help='merged_submit.csv')        
        parser.add_argument('--img', required=False,
                        metavar="/path/to/dataset/",    default = None,
                        help='Load image to crop objs , not required')
    args = parser.parse_args()    

    ###   running
    os.makedirs(args.output_dir,exist_ok=True)

    if args.command == "iou":   
        '''compare iou of 2 segemtnation result from 2 full label image'''
        if os.path.isfile(args.pd) :  # one big file
            hf = h5py.File(args.pd, 'r')                                                             # load wholelabel use 9s
            pd_labels = np.array(hf.get('seg_results'))
            hf.close()
            hf = h5py.File(args.gt, 'r')                                                             # load wholelabel use 9s
            gt_labels = np.array(hf.get('seg_results'))
            hf.close()

            mIoU,good_iou_rate,excellent_iou_rate,iou_df = iou_labels_calculation(gt_labels, pd_labels)
            iou_df.to_csv (os.path.join(args.output_dir,"iou_df.csv"))
            print("mIoU=", mIoU, "good_iou_rate = ", good_iou_rate, "excellent_iou_rate=", excellent_iou_rate )
        
        else:                    # folders contain files
            
            imgid_ls =  []
            pd_mask = gt_mask =[]
            max_pd =  max_gt =0
            ii = 0 
            for image_mask in os.listdir(args.pd):
                if "_mask.png" in image_mask and "color" not in image_mask and "watershed" not in image_mask  :                
                    if image_mask in os.listdir(args.gt):                    
                        print (image_mask)
                        imgid_ls.append(image_mask)
                        pd_mask_img = io.imread (os.path.join (args.pd,image_mask))[:,:,0]
                        gt_mask_img = io.imread (os.path.join (args.gt,image_mask))[:,:,0]                        
                        __,__,__,pd_labels = dt_utils.mask2props(pd_mask_img)
                        __,__,__,gt_labels = dt_utils.mask2props(gt_mask_img)
                        pd_labels = pd_labels + max_pd
                        gt_labels = gt_labels + max_gt
                        
                        max_pd = pd_mask_img.max() + 1
                        max_gt = gt_mask_img.max() + 1

                        __,__,__,iou_df_crop = iou_labels_calculation(gt_labels, pd_labels)

                        if ii == 0 :
                            iou_df = iou_df_crop
                        else:                            
                            iou_df = pd.concat([iou_df, iou_df_crop], ignore_index=True)
                        ii += 1
        plt.figure()
        plt.hist(iou_df["iou_ls"],bins=np.arange(0, 1.1, 0.01) )
        plt.savefig(os.path.join(args.output_dir,"iou_obj_hist.png"))

    elif args.command == "iou_ann":  
        '''compare segemntation result with crops of annotated ground truth'''
        # load merged csv
        # sys.path.insert(0, os.path.join(ROOT_DIR,"mrcnn"))
        # import utils
        # generate the iou compare result for two image set

        submit_path = args.submit
        if submit_path is not None:            
            if os.path.isdir( args.submit):   # # look for the latest submit.csv
                submit_path = None
                for f in os.listdir(args.submit):
                    if "submit_" in f:
                        submit_temp = os.path.join(args.submit,f,"submit.csv")
                        if os.path.isfile(submit_temp):
                            submit_path = submit_temp
            elif os.path.isfile( args.submit):
                submit_path = args.submit
        else:
            submit_path = None
                
        atlas_df = iou_ann(args.pd, args.gt, args.output_dir,submit_df= submit_path, 
                            mp = dt_utils.str2bool(args.mp) ,error_obj_path= args.error_obj_path)       
        print("eval_gt save in ",args.output_dir)
    elif args.command == "iou_map":  
        atlas_df =  pd.read_csv(args.submit) 
        draw_ioufmap(atlas_df, args.output_dir)