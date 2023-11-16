# Author: Xinyi Wang
# Date: 2021/08/31

import numpy as np
import pandas
import seaborn as sn
import matplotlib.pyplot as plt
import h5py

data_name = 'CNN_features_360P'
csv_file = '/mnt/storage/home/um20242/scratch/BVQA_2023/mos_file/YOUTUBE_UGC_360P_metadata.csv'
features_dir = '/mnt/storage/home/um20242/scratch/VSFA-UGC/CNN_features_360P/'

# data_name = 'CNN_features_KoNVid'
# csv_file = '/mnt/storage/home/um20242/scratch/BVQA_2023/mos_file/KONVID_1K_metadata.csv'
# features_dir = '/mnt/storage/home/um20242/scratch/VSFA-UGC/CNN_features_KoNVid/'

try:
    df = pandas.read_csv(csv_file)
except:
    raise Exception('Read csv file error!')

names = df['vid']
# names = df['flickr_id']
features = []
score = []
for i in range(len(df)):
    feature = np.load(features_dir + str(i) + '_resnet-50_res5c.npy', allow_pickle=True)
    mos = np.load(features_dir + str(i) + '_score.npy', allow_pickle=True)

    median_feature = np.median(feature, axis=0)
    feature_list = median_feature.tolist()

    features.append(feature_list)
    score.append(mos)

cnnfeatures = pandas.DataFrame(index=names, data=features)
score = pandas.DataFrame(index=names, data=score)
print(score)
print(cnnfeatures)

cnnfeatures.to_csv('/mnt/storage/home/um20242/scratch/VSFA-UGC/CNN_features/' + data_name + '_median.csv')
score.to_csv('/mnt/storage/home/um20242/scratch/VSFA-UGC/CNN_features/' + data_name + '_median_score.csv')


# print(len(result1))
# print(result1.shape)
# # print(results.ndim)
# # print(type(results))
# # print(results.dtype)
# # print(results.size)
# print(result2.shape)
# re1 = result1.flatten()
# print(re1.shape)
# re2 =result2.flatten()
# print(re2.shape)

