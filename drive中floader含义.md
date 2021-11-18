drive/MyDrive/Meta-baseline

- 2021Y-11M-17D-07H-15Minu

  - Resnet最后一层加上CBAM，运行train_classifier.py得到的
  - /content/baseline/meta-baseline-CBAM/save/classifier_mini-imagenet_resnet12-cbam
  - epoch 100, train 0.6747|0.8126, val 0.7680|0.7890, fs 1: 0.6196 5: 0.7857, 2.9m 2.8h/2.8h
  - 观察tensorboard，classifer的loss下降趋势仍然剧烈，说明有调参空间

- 2021Y-11M-17D-11H-03Minu

  - Resnet最后一层加上CBAM，运行train_classifier.py得到的resnet，运行train_meta.py 20个epoch得到的

  - /content/baseline/meta-baseline-CBAM/save/meta_mini-imagenet-1shot_meta-baseline-resnet12-cbam

  - ```
    num params: 8.0M
    test epoch 1: acc=63.22 +- 0.75 (%), loss=0.9276 (@7)
    test epoch 2: acc=63.20 +- 0.54 (%), loss=0.9263 (@2)
    test epoch 3: acc=63.25 +- 0.44 (%), loss=0.9264 (@0)
    test epoch 4: acc=63.17 +- 0.38 (%), loss=0.9265 (@18)
    test epoch 5: acc=63.23 +- 0.34 (%), loss=0.9242 (@6)
    test epoch 6: acc=63.29 +- 0.31 (%), loss=0.9214 (@16)
    test epoch 7: acc=63.26 +- 0.28 (%), loss=0.9216 (@13)
    test epoch 8: acc=63.28 +- 0.26 (%), loss=0.9202 (@0)
    test epoch 9: acc=63.16 +- 0.25 (%), loss=0.9220 (@10)
    test epoch 10: acc=63.16 +- 0.23 (%), loss=0.9218 (@9)
    ```

  - 观察tensorboard，train-meta的loss/train稳定下降，但loss/tval、loss/val几乎不变，acc也几乎不变

    - ```python
      epoch 1, train 0.5403|0.8269, tval 0.9385|0.6297, val 0.9382|0.6364, 3.3m 3.3m/1.1h (@7)
      epoch 2, train 0.4806|0.8499, tval 0.9215|0.6364, val 0.9202|0.6445, 3.3m 6.6m/1.1h (@7)
      epoch 3, train 0.4495|0.8624, tval 0.9141|0.6392, val 0.9093|0.6484, 3.3m 9.9m/1.1h (@7)
      epoch 4, train 0.4189|0.8712, tval 0.9088|0.6414, val 0.9044|0.6487, 3.3m 13.2m/1.1h (@7)
      epoch 5, train 0.4078|0.8781, tval 0.9103|0.6393, val 0.9030|0.6498, 3.3m 16.6m/1.1h (@7)
      epoch 6, train 0.3872|0.8861, tval 0.9103|0.6384, val 0.8968|0.6511, 3.3m 19.9m/1.1h (@7)
      epoch 7, train 0.3839|0.8852, tval 0.9072|0.6400, val 0.8914|0.6537, 3.3m 23.2m/1.1h (@7)
      epoch 8, train 0.3578|0.8952, tval 0.9042|0.6404, val 0.8889|0.6554, 3.3m 26.5m/1.1h (@7)
      epoch 9, train 0.3459|0.8994, tval 0.9085|0.6397, val 0.8888|0.6542, 3.3m 29.8m/1.1h (@7)
      epoch 10, train 0.3405|0.8999, tval 0.9089|0.6396, val 0.8840|0.6568, 3.3m 33.1m/1.1h (@7)
      epoch 11, train 0.3373|0.9000, tval 0.9108|0.6375, val 0.8844|0.6550, 3.3m 36.4m/1.1h (@7)
      epoch 12, train 0.3217|0.9061, tval 0.9118|0.6389, val 0.8831|0.6572, 3.3m 39.7m/1.1h (@7)
      epoch 13, train 0.3051|0.9125, tval 0.9117|0.6374, val 0.8830|0.6567, 3.3m 43.0m/1.1h (@7)
      epoch 14, train 0.3065|0.9120, tval 0.9175|0.6353, val 0.8832|0.6570, 3.3m 46.3m/1.1h (@7)
      epoch 15, train 0.2957|0.9154, tval 0.9147|0.6366, val 0.8869|0.6550, 3.3m 49.6m/1.1h (@7)
      epoch 16, train 0.2907|0.9181, tval 0.9180|0.6356, val 0.8869|0.6552, 3.3m 52.9m/1.1h (@7)
      epoch 17, train 0.2863|0.9175, tval 0.9189|0.6347, val 0.8853|0.6557, 3.3m 56.2m/1.1h (@7)
      epoch 18, train 0.2780|0.9219, tval 0.9170|0.6353, val 0.8890|0.6542, 3.3m 59.4m/1.1h (@7)
      epoch 19, train 0.2784|0.9196, tval 0.9245|0.6331, val 0.8869|0.6557, 3.3m 1.0h/1.1h (@7)
      epoch 20, train 0.2617|0.9269, tval 0.9276|0.6322, val 0.8861|0.6560, 3.3m 1.1h/1.1h (@7)
      ```

    - 说明train-meta阶段增加epoch没有意义

