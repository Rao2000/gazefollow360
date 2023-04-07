# Looking here or there? Gaze Following in 360-Degree Images

```
# train the gde module firstly
python train.py
# train the dp module after loading pretrained gde model
python train_dp.py 
# train the df module after loading pretrained gde and dp model
python train_all.py
```

Our experimental results are as follows. At the same time, we provide the pretrained models.

|              | pixel dist | norm dist | AUC    | sphere dist |
| ------------ | ---------- | --------- | ------ | ----------- |
| test dataset | 443.5346   | 0.1584    | 0.8673 | 0.6157      |

