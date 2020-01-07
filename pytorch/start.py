from __future__ import print_function
import torch

x = torch.empty(5, 3)
print(x)

y = torch.zeros(5, 3, dtype=torch.long)
print(y)

z = torch.tensor([5.5, 3.3])
print(z)

a = torch.randn_like(x, dtype=torch.float)
print(a)
print(a.size())

x1 = torch.ones(5, 3)
x2 = x1.new_ones([5, 3])
x3 = x1 + x2

result = torch.empty([5, 3])
print(x3, torch.add(x1, x3, out=result))
print(result)

print(x3)
x3.add_(x1)
print(x3)

x = torch.randn(4, 4)
y = x.view(2, 8)
z = x.view(-1, 8)
print(x.size(), y.size(), z.size())

x = torch.randn(1)
print(x.item())

print(torch.randn(1, 10))
print(torch.rand(1, 10))

print(x)
print(x.numpy())
