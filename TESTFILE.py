import torch

a=torch.randn([2,3,4,4])
b=torch.randn([2,3,4,4])

c=torch.cat([a,b],dim=0)
print(c.size())
