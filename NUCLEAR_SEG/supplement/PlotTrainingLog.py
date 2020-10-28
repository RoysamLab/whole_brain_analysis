'''
 plot training runtime loss from Log file 
Author: Rebecca LI, University of Houston, Farsight Lab, 2018
xiaoyang.rebecca.li@gmail.com


'''


from math import sqrt
import os,sys,time
import pandas as pd
import numpy as np

def createDataFrame(Log_fileName,epoch = 40):
    f = open(Log_fileName,"r")

    log_df = pd.DataFrame(np.zeros([1,7]), columns = ["Epoch","loss","mrcnn_bbox_loss","mrcnn_class_loss","mrcnn_mask_loss","rpn_bbox_loss","rpn_class_loss"])
    logTol_df = pd.DataFrame(np.zeros([1,8]), columns = ["Epoch","time","val_loss","val_rpn_class_loss","val_rpn_bbox_loss", "val_mrcnn_class_loss","val_mrcnn_bbox_loss","val_mrcnn_mask_loss"])

    Epoch = {}
    for Epoch_id in range(1,epoch +1):
        Epoch[ "Epoch " + str(Epoch_id) + "/40" ] = []  

    for l_id , line in enumerate(f):   
        for Epoch_id in Epoch: 
            if Epoch_id in line:   
                current_Epoch = Epoch_id
                Epoch[current_Epoch] ={}

        if "] -" in line:
            it2itTotal = line.split("[")[0]
            measures   = line.split("] - ")[1]    
            it         = int(it2itTotal.split("/")[0])
            itTotal    = int(it2itTotal.split("/")[1])       
        #     print ("it=", it , "itTotal=",itTotal)
            # 1/43 ...43/43
            measure_dict = {}

            Epoch_id = int( current_Epoch.split("/")[0].split(" ")[1])
            measure_dict["Epoch"] = Epoch_id
            measures_disp = measures.split("-")
            for measure_disp in measures_disp:
                if "ETA" not in measure_disp:
                    measure_name = measure_disp.split(":")[0].split(" ")[1]
                    if measure_name in log_df.columns.values.tolist():
                        measure_value = float(measure_disp.split(":")[1])          
                        measure_dict[measure_name] = measure_value

            log_df = log_df.append(measure_dict , ignore_index=True)

            # 43/43
            if it == itTotal:   
                measures_disp = measures.split("-")
                measure_dict["Epoch"] = Epoch_id
                measures_disp = measures.split("-")

                for measure_disp in measures_disp:
                    measure_name = measure_disp.split(":")[0].split(" ")[1]
                    if 'step' in measure_disp:
                        measure_dict["time"] = int(measure_disp.split("s")[0])
                    if measure_name in logTol_df.columns.values.tolist():
                        measure_value = float(measure_disp.split(":")[1])        
                        measure_dict[measure_name] = measure_value
                logTol_df = logTol_df.append(measure_dict , ignore_index=True)

    return log_df,logTol_df

def plot_log(log_df,logTol_df,savePath):
    if os.path.exists(savePath) is False:
        os.mkdirs(savePath)
    plt.figure(figsize=(10,12),dpi=100)
    plt.suptitle("Performance in Batches")
    for i,measure in enumerate( ["loss","mrcnn_bbox_loss","mrcnn_class_loss",
                                "mrcnn_mask_loss","rpn_bbox_loss","rpn_class_loss"]):
        plt.subplot(3,2,i+1)
        plt.plot(list(log_df.index)[1:],log_df[measure][1:])
        plt.title(measure)
        plt.xlabel("Batch")
        plt.ylabel(measure)
    plt.tight_layout()
    plt.savefig(os.path.join( savePath, "Performance in Batches.tif"))

    plt.figure(figsize=(10,20),dpi=100)
    for i,measure in enumerate( ["time","val_loss","val_rpn_class_loss","val_rpn_bbox_loss", "val_mrcnn_class_loss","val_mrcnn_bbox_loss","val_mrcnn_mask_loss"]):
        plt.subplot(4,2,i+1)
        plt.plot(list(logTol_df["Epoch"])[1:],logTol_df[measure][1:])
        plt.title(measure)
        plt.xlabel("Epoch")
        plt.ylabel(measure)
    plt.tight_layout()
    plt.suptitle("Performance in Epochs")
    plt.savefig(os.path.join( savePath,"Performance in Epochs.tif"))
  
if __name__ == '__main__':
    import matplotlib
    # Agg backend runs without a display
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='****  plot training runtime loss from Log file ******' ,
        formatter_class=argparse.RawTextHelpFormatter)
        
       
    parser.add_argument('-l','--log', required=False,
                        metavar="/path/to/logFile/",default = None,
                        help='Log file when training')   
    parser.add_argument('-r','--results', required=False,
                        metavar="/path/to/result/",default = None,
                        help='directory for results')
    args = parser.parse_args()

    tic = time.time()

    log_df,logTol_df = createDataFrame(args.log)
    plot_log(log_df,logTol_df,args.results)

    toc2 =  time.time()
    print ("total time = ", int(toc2 - tic))