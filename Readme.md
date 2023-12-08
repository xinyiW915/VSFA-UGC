# UGC video dataset test based on VSFA

## UPDATE!
Completed implementation for 360P, 480P, 720P, 1080P video data on YouTube UGC datasets.

Scatter plots and nonlinear logistic fitted curves of VSFA model versus MOS trained with a grid-search SVR using k-fold cross-validation on YouTube UGC datasets.

The 1080P dataset was split for training due to lack of memory. The results may therefore be biased and will continue to be checked for related issues.

## Description
VSFA code based on the following papers:

- Dingquan Li, Tingting Jiang, and Ming Jiang. [Quality Assessment of In-the-Wild Videos](https://dl.acm.org/citation.cfm?doid=3343031.3351028). In Proceedings of the 27th ACM International Conference on Multimedia (MM ’19), October 21-25, 2019, Nice, France. [[arxiv version]](https://arxiv.org/abs/1908.00375)
- This project is based on [lidq92/VSFA]([https://github.com/vztu/RAPIQUE](https://github.com/lidq92/VSFA))

### Intra-Database Experiments (Training and Evaluating)
#### Feature extraction

```
CUDA_VISIBLE_DEVICES=0 python CNNfeatures.py --database=KoNViD-1k --frame_batch_size=64
```

You need to specify the `database` and change the corresponding `videos_dir`.

#### Quality prediction

```
CUDA_VISIBLE_DEVICES=0 python VSFA.py --database=KoNViD-1k --exp_id=0
```

You need to specify the `database` and `exp_id`.

#### Visualization
```bash
tensorboard --logdir=logs --port=6006 # in the server (host:port)
ssh -p port -L 6006:localhost:6006 user@host # in your PC. See the visualization in your PC
```

### Test Demo

The model weights provided in `models/VSFA.pt` are the saved weights when running the 9-th split of KoNViD-1k.
```
python test_demo.py --video_path=test.mp4
```

### Requirement
```bash
conda create -n reproducibleresearch pip python=3.6
source activate reproducibleresearch
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
source deactive
```
- PyTorch 1.1.0
- TensorboardX 1.2, TensorFlow-TensorBoard

Note: The codes can also be directly run on PyTorch 1.3.

