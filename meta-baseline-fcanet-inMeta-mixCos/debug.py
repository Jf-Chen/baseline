
in meta_baseline.py
line 133

print("base",base.size(),"query",query.size())
base torch.Size([4, 5, 1, 640, 5, 5]) query torch.Size([4, 75, 640, 5, 5])

print("base_mix",base_mix.size(),"query_mix",query_mix.size())
base_mix torch.Size([4, 5, 640, 25]) query_mix torch.Size([4, 75, 25, 640])

print(innerproduct_matrix.size())
torch.Size([25, 25])

print("topk_value",topk_value.size(),"topk_index", topk_index.size() )
topk_value torch.Size([25, 5]) topk_index torch.Size([25, 5])

print(Similarity_list.size())
torch.Size([4, 5])

print(logits_dn4.size())
print(logits.size())

print(inner_sim.size())
torch.Size([1, 5])

print(num_q)
75

print(logits_dn4[0,0,:])
tensor([44.7460, 28.7631, 30.3936, 43.2586, 36.3214], device='cuda:0',
       grad_fn=<SliceBackward0>)
print(logits[0,0,:])
tensor([0.5367, 0.3266, 0.3297, 0.5300, 0.3876], device='cuda:0',
       grad_fn=<SliceBackward0>)



