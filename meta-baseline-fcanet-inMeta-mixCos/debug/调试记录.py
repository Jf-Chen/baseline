print(self.att)
(Pdb) print(self.att)
MultiSpectralAttentionLayer(
  (dct_layer): MultiSpectralDCTLayer()
  (fc): Sequential(
    (0): Linear(in_features=640, out_features=40, bias=False)
    (1): ReLU(inplace=True)
    (2): Linear(in_features=40, out_features=640, bias=False)
    (3): Sigmoid()
  )
)

print(encoder)


print(x.size())
torch.Size([128, 640, 5, 5])
torch.Size([128, 640])

print("x_shot",x_shot.size(),"\n","x_shot_att",x_shot_att.size(),"\n",
"x_shot_aft",x_shot_aft.size(),"\n","x_shot_pool",x_shot_pool.size())
 x_shot torch.Size([20, 640, 5, 5]) 
 x_shot_att torch.Size([20, 640, 5, 5]) 
 x_shot_aft torch.Size([4, 5, 1, 640, 5, 5]) 
 x_shot_pool torch.Size([4, 5])

print("x_shot",x_shot.size(),"\n","x_shot_att",x_shot_att.size(),"\n",
"x_shot_aft",x_shot_aft.size(),"\n","x_shot_pool",x_shot_pool.size())
x_shot torch.Size([20, 640, 5, 5]) 
 x_shot_att torch.Size([20, 640, 5, 5]) 
 x_shot_aft torch.Size([4, 5, 1, 640, 5, 5]) 
 x_shot_pool torch.Size([4, 5, 1, 640])
 
 print("x_shot_F",x_shot_F.size(),"x_query_F",x_query_F.size())
 x_shot_F torch.Size([4, 5, 640]) x_query_F torch.Size([4, 75, 640, 5])
 
 print("x_query_aft",x_query_aft.size(),"x_query_pool",x_query_pool.size())
 x_query_aft torch.Size([4, 75, 640, 5, 5]) x_query_pool torch.Size([4, 75, 640, 5])