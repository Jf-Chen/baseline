print(config['model'],config['model_args'])
(Pdb) print(config['model'],config['model_args'])
meta-baseline-out {'encoder': 'resnet12-wide_without_avgpool', 'encoder_args': {}}

print(models.name)
models.make('meta-baseline-out',{})
model_temp = models.make('meta-baseline', **config['model_args'])
print(model_temp)

model_temp_2 = models.make('meta-baseline-out', **config['model_args'])

print("shot_shape",shot_shape,"query_shape",query_shape,"img_shape",img_shape)
shot_shape torch.Size([4, 5, 1]) query_shape torch.Size([4, 75]) img_shape torch.Size([3, 80, 80])
print("x_shot",x_shot.size(),"x_query",x_query.size(),"x_tot",x_tot.size())
x_shot torch.Size([4, 5, 1, 16000]) x_query torch.Size([4, 75, 16000]) x_tot torch.Size([320, 640, 5, 5])

def forward(self, x_shot, x_query):
        shot_shape = x_shot.shape[:-3]
        query_shape = x_query.shape[:-3]
        img_shape = x_shot.shape[-3:]

        x_shot = x_shot.view(-1, *img_shape)
        x_query = x_query.view(-1, *img_shape)
        x_tot = self.encoder(torch.cat([x_shot, x_query], dim=0))
        x_shot, x_query = x_tot[:len(x_shot)], x_tot[-len(x_query):]
        x_shot = x_shot.view(*shot_shape, -1)
        x_query = x_query.view(*query_shape, -1)

print("x_shot",x_shot.size(),"x_query",x_query.size())
(Pdb) print("x_shot",x_shot.size(),"x_query",x_query.size())
x_shot torch.Size([4, 5, 1, 3, 80, 80]) x_query torch.Size([4, 75, 3, 80, 80])

print("shot_shape",shot_shape,"query_shape",query_shape,"img_shape",img_shape)
shot_shape torch.Size([4, 5, 1]) query_shape torch.Size([4, 75]) img_shape torch.Size([3, 80, 80])

print(x_shot.view(-1, *img_shape).size())
(Pdb) print(x_shot.view(-1, *img_shape).size())
torch.Size([20, 3, 80, 80])

print(x_query.view(-1, *img_shape).size())
(Pdb) print(x_query.view(-1, *img_shape).size())
torch.Size([300, 3, 80, 80])

x_tot = self.encoder(torch.cat([x_shot, x_query], dim=0))
print("x_tot",x_tot.size())
(Pdb) print("x_tot",x_tot.size())
x_tot torch.Size([320, 640, 5, 5])

(Pdb) print("x_shot",x_shot.size(),"x_query",x_query.size())
x_shot torch.Size([20, 640, 5, 5]) x_query torch.Size([300, 640, 5, 5])

(Pdb) print("x_shot",x_shot.size(),"x_query",x_query.size())
x_shot torch.Size([4, 5, 1, 16000]) x_query torch.Size([4, 75, 16000])

print(logits.size())
(Pdb) print(logits.size())
torch.Size([4, 75, 5])




























