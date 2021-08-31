import numpy as np
import pandas
import seaborn as sn
import matplotlib.pyplot as plt
import h5py

data_name = 'CNN_features_1080P'
csv_file = '/mnt/storage/home/um20242/scratch/ugc-dataset/data_filter/YOUTUBE_UGC_1080P_metadata.csv'
features_dir = '../' + data_name + '/'

try:
    df = pandas.read_csv(csv_file)
except:
    raise Exception('Read csv file error!')

names = df['vid']
features = []
score = []
for i in range(len(df)):
    feature = np.load(features_dir + str(i) + '_resnet-50_res5c.npy', allow_pickle=True)
    mos = np.load(features_dir + str(i) + '_score.npy', allow_pickle=True)

    mean_feature = np.average(feature, axis=0)
    feature_list = mean_feature.tolist()

    score.append(mos)
    features.append(feature_list)

cnnfeatures = pandas.DataFrame(index=names, data=features)
score = pandas.DataFrame(index=names, data=score)
print(score)
print(cnnfeatures)

cnnfeatures.to_csv('../CNN_features/' + data_name + '.csv')
score.to_csv('../CNN_features/' + data_name + '_score.csv')


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

