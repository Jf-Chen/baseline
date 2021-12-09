print("x_shot_aft",x_shot_aft.size(),"x_query_aft",x_query_aft.size())
x_shot_aft torch.Size([4, 5, 5, 640, 5, 5]) x_query_aft torch.Size([4, 75, 640, 5, 5])

print("base_mean",base_mean.size(),"query_mean",query_mean.size())
base_mean torch.Size([4, 25, 640]) query_mean torch.Size([4, 75, 640])

print("logits_cos",logits_cos.size())
logits_cos torch.Size([4, 75, 25])

print("query_mix",query_mix.size(),"base_mix",base_mix.size())
query_mix torch.Size([4, 75, 25, 640]) base_mix torch.Size([4, 5, 3200, 25])

print("base_temp_1",base_temp_1.size())
base_temp_1 torch.Size([4, 5, 5, 640, 25])



