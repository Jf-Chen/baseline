print("eval_fs==",eval_fs)
(Pdb) print("eval_fs==",eval_fs)
eval_fs== True

print(model)
(Pdb) print(model)
Classifier(
  (encoder): ResNet12(
    (layer1): Block(
  )
  (classifier): LinearClassifier(
    (linear): Linear(in_features=512, out_features=64, bias=True)
  )
)


print(x.size())
(Pdb) print(x.size())
torch.Size([128, 512, 5, 5])

print(self.encoder)
(Pdb) print(self.encoder)
ResNet12(
  (layer1): Block(
  
x = self.encoder(x)
(Pdb) print(x.size())
torch.Size([128, 3, 80, 80])

def forward(self, x):
    print("in classifer.py Classifer().forward line 21")
    pdb.set_trace() # [128, 3, 80, 80]
    x = self.encoder(x)
    print("in classifer.py Classifer().forward line 25")
    pdb.set_trace() # [128, 512, 5, 5]
    x = self.classifier(x)
    print("in classifer.py Classifer().forward line 28")
    pdb.set_trace() # 形状不匹配
    return x
    

print("encoder==",encoder,"classifier==",classifier)
(Pdb) print("encoder==",encoder,"classifier==",classifier)
encoder== resnet12 classifier== linear-classifier

print("encoder==",encoder,"classifier==",classifier,"encoder_args",encoder_args)
print("encoder_args",encoder_args)
print("classifier_args",classifier_args)

print("encoder.out_dim",encoder.out_dim)
print("encoder==",encoder)

假如ResNet12是带平均池化的，forward的过程应该是：
[128, 3, 80, 80]
[128, 512]
[128, 64]
不过现在去掉池化后是
[128, 3, 80, 80]
[128, 512, 5, 5]
需要最后变成
[128, 64]
所以linear-classifier要改下，resnet12的属性outdim也要改下
resnet12需要加上resolution属性


print("channels == ",channels)
print("self.layer4",self.layer4)





