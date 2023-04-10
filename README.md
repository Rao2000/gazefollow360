# Looking here or there? Gaze Following in 360-Degree Images

# How to use

```
# train the gde module firstly
python train.py
# train the dp module after loading pretrained gde model
python train_dp.py 
# train the df module after loading pretrained gde and dp model
python train_all.py
```

# Model

paper ：Looking here or there? Gaze Following in 360-Degree Images

[paper link](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_Looking_Here_or_There_Gaze_Following_in_360-Degree_Images_ICCV_2021_paper.pdf)

github link：[Rao2000/gazefollow360: The paper &#34;Looking here or there? Gaze Following in 360-Degree Images&#34; implementing with Pytorch (github.com)](https://github.com/Rao2000/gazefollow360)

In this paper, a new task of gaze following in 360-degree images was studied, and a new large-scale dataset named "GazeFollow360" was collected for this task. The paper pointed out that the main challenge of this new task is the distortion caused by the projection of the spherical image onto a flat plane and the mismatch between the gaze target and the gaze direction of human subjects. To address this problem, a dual-path framework was proposed, which models gaze guidance in 3D spherical space instead of simply detecting gaze targets in parallel in local and distant regions on the 2D image plane. The strong performance of the proposed framework on GazeFollow360 validates its potential in understanding gaze behavior in the real 3D world.

The main contributions can be divided into three parts:

* The first work on gaze following in 360-degree unwrapped images.
* Establishing a dataset of 10,058 images.
* Proposing a dual-path framework to solve gaze following in 360-degree scenes.

![1681109252269](image/README/1681109252269.png)


# Result

Experimental results in paper:

|              |  AUC  | sphere dist |
| :----------: | :----: | :---------: |
| test dataset | 0.9350 |   0.6067   |


Our experimental results are as follows. At the same time, we provide the pretrained models.

|              | pixel dist | norm dist | AUC    | sphere dist |
| ------------ | ---------- | --------- | ------ | ----------- |
| test dataset | 443.5346   | 0.1584    | 0.8673 | 0.6157      |

# Other links

Chinese Readme : [link1](https://zhuanlan.zhihu.com/p/620706533)  [link2](https://blog.csdn.net/weixin_42264234/article/details/130056365) [link3](https://juejin.cn/post/7220244993788493861)

GazeOnce Code: [link]()
