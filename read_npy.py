import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

# np.save(save_result_file, (y_pred, y_test, test_loss, SROCC, KROCC, PLCC, RMSE, test_index))
results = np.load('/mnt/storage/home/um20242/scratch/VSFA-master/CNN_features_2160P/2_score.npy',allow_pickle=True)

print(results)


