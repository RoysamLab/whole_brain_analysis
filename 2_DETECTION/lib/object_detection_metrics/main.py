import pandas as pd
import numpy as np

bbxs_detection = pd.read_csv('bbxs_detection.txt', sep='\t')
bbxs_detection = bbxs_detection[['xmin', 'ymin', 'xmax', 'ymax']].values
labels = np.expand_dims(['Nucleus'] * bbxs_detection.shape[0], axis=1)
scores = np.expand_dims(np.load('scores.npy'), axis=1)
detection = np.hstack((labels, scores, bbxs_detection))

bbxs_validation = pd.read_csv('bbxs_validation.txt', sep='\t')
bbxs_validation = bbxs_validation[['xmin', 'ymin', 'xmax', 'ymax']].values
labels = np.expand_dims(['Nucleus'] * bbxs_validation.shape[0], axis=1)
validation = np.hstack((labels, bbxs_validation))



pd.DataFrame(detection).to_csv('detection/2.txt', sep=' ', index=False, header=None)
pd.DataFrame(validation).to_csv('validation/2.txt', sep=' ', index=False, header=None)

a = 1
