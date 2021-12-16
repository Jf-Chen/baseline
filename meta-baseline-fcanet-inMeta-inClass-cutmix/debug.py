print("x_shot_aft",x_shot_aft.size(),"x_query_aft",x_query_aft.size())
x_shot_aft torch.Size([4, 5, 5, 640, 5, 5]) x_query_aft torch.Size([4, 75, 640, 5, 5])

print("base_mean",base_mean.size(),"query_mean",query_mean.size())
base_mean torch.Size([4, 25, 640]) query_mean torch.Size([4, 75, 640])

print("logits_cos",logits_cos.size())
logits_cos torch.Size([4, 75, 25])

print("query_mix",query_mix.size(),"base_mix",base_mix.size())
query_mix torch.Size([4, 75, 25, 640]) base_mix torch.Size([4, 5, 3200, 25])
query_mix torch.Size([4, 75, 25, 640]) base_mix torch.Size([4, 5, 125, 640])

print("base_temp_1",base_temp_1.size())
base_temp_1 torch.Size([4, 5, 5, 640, 25])

query_sam@support_set_sam
print("query_sam",query_sam.size(),"support_set_sam",support_set_sam.size())
query_sam torch.Size([25, 640]) support_set_sam torch.Size([125, 640])

temp=query_sam@torch.transpose(support_set_sam,0,1)

print("logits_dn4",logits_dn4.size(),"logits_cos",logits_cos.size())
logits_dn4 torch.Size([4, 75, 5]) logits_cos torch.Size([4, 75, 25])

print("label",label.size())
label torch.Size([300])
temp= F.cross_entropy(logits_cos, label)

print("logits_dn4_view",logits_dn4_view.size(),"logits_cos_view",logits_cos_view.size())
logits_dn4_view torch.Size([300, 5]) logits_cos_view torch.Size([300, 5])

print("loss_dn4",loss_dn4.size(),"loss_cos",loss_cos.size())

print(logits_dn4[0,0,:])

print(logits_cos[0,0,:])

print(loss_dn4,loss_cos)

print("logits_dn4[0,0,:]",logits_dn4[0,0,:],"logits_dn4_norm[0,0,:]",logits_dn4_norm[0,0,:])
logits_dn4[0,0,:] tensor([30.7755, 20.6427, 26.2384, 21.6074, 23.1187], device='cuda:0',
       grad_fn=<SliceBackward0>) logits_dn4_norm[0,0,:] tensor([0.5561, 0.3730, 0.4741, 0.3904, 0.4177], device='cuda:0',
       grad_fn=<SliceBackward0>)

print("logits_cos[0,0,:]",logits_cos[0,0,:],"logits_cos_norm[0,0,:]",logits_cos_norm[0,0,:])
logits_cos[0,0,:] tensor([0.6816, 0.4036, 0.4062, 0.5474, 0.5820], device='cuda:0',
       grad_fn=<SliceBackward0>) logits_cos_norm[0,0,:] tensor([0.5698, 0.3374, 0.3396, 0.4576, 0.4865], device='cuda:0',
       grad_fn=<SliceBackward0>)

print("lable",label.size(),"target_a",target_a.size(),"target_b",target_b.size())

lable torch.Size([150]) target_a torch.Size([150]) target_b torch.Size([2])

ValueError: Expected input batch_size (150) to match target batch_size (2).

print("data",data.size(),"label",label.size())
data torch.Size([128, 3, 80, 80]) label torch.Size([128])

print("target_a",target_a.size(),"target_b",target_b.size())
target_a torch.Size([128]) target_b torch.Size([128])




















