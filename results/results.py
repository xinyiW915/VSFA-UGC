import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

# np.save(save_result_file, (y_pred, y_test, test_loss, SROCC, KROCC, PLCC, RMSE, test_index))
results = np.load('/mnt/storage/home/um20242/scratch/VSFA-master/results/VSFA-YOUTUBE_UGC_480P-EXP0.npy',allow_pickle=True)

y_pred = results[0]
y_test = results[1]

test_loss = results[2]
print('test_loss:', test_loss)
SRCC = results[3]
print('SRCC:', SRCC)
KRCC = results[4]
print('KRCC:', KRCC)
PLCC = results[5]
print('PLCC:', PLCC)
RMSE = results[6]
print('RMSE:', RMSE)
test_index = results[7]
print('test_index:', test_index)

# # 画图
data_name = 'YOUTUBE_UGC_480P'
algo_name = 'VSFA'
fig_path = '/mnt/storage/home/um20242/scratch/VSFA-master/figs/'


data = {'MOS': y_test,
        'Predicted Score': y_pred}
        #'Predicted Score': y_pred_logistic}
df = pd.DataFrame(data)
print(df)
fig = sn.regplot(x='Predicted Score', y='MOS', data=df, marker='o',
                order = 2,  # 默认为1，越大越弯曲
                scatter_kws={'color': '#016392', },  # 设置散点属性，参考plt.scatter
                line_kws={'linestyle': '--', 'color': '#c72e29'}  # 设置线属性，参考 plt.plot#
                )
plt.show()
plt.title("480P YT-UGC", fontsize=10)
reg_fig = fig.get_figure()
reg_fig.savefig(fig_path + algo_name + '_' + data_name, dpi=400)

