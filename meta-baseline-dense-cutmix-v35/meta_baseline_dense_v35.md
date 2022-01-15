时间 2022/01/14 星期四

#### ipynb



#### location 



#### 方法主体

#### 结论

1. cutmix在pretrain阶段没有效果，
2. pretrain阶段的参数最好是[70,90]，甚至[90]+0.1->0.001
3. meta-train阶段的参数就保持0.001

#### 数据集

miniImagenet

"/content/drive/MyDrive/few_shot_meta_baseline/materials/miniImageNet.zip"

###  参数

```
classifier: 
	sgd,
	max_epoch:100
	lr = 0.1
	MultiStepLR
	gamma = 0.1
	milestone = [90]
meta:
	sgd
	max_epoch:20
	lr = 0.001
	MultiStepLR
	milestone = [90]
```



#### train_classifier.py

- ```
  
  ```
- 

#### train_meta.py

- 5-way 5-shot
  - ```
    
    ```
  - ```
    
    ```

#### few_shot_test.py测试结果

- 5-way 1-shot的meta-train测试

  - load: 
  
  - ```shell
  
    ```
    
  - ```
    
    ```
  
- 5-way 5-shot 的meta-train测试

  - load: 
  
  - ```
  
    ```
    
  - ```
    
    ```

#### 评价

- pre-train阶段，lr=0.1->0.01，meta阶段，lr=0.001不变，就可以取得不错的效果
  - 实验是
  - 论文是

#### tensorboard

- ```python
  
  ```
  
- classifier
  
  -      
- meta 1-shot
  
  - ​       
- meta 5-shot
  
  -       
- 分析
  - 
  - 

  
  