import torch

from Transforms.Lorentz import LorentzRelativity
from utils import *


lorentz = LorentzRelativity()

d_1 = SPEED_OF_LIGHT * torch.tensor([8.0])
t_1 = torch.tensor([10.0])
u_1 = 0.8 * SPEED_OF_LIGHT * torch.tensor([1.0])
res_1d = lorentz.compute_moving(d_1, t_1, u_1)

print("Computation in 1D")
print("Observer Position", formatResult(res_1d['d_observer']))
print("Observer Time", res_1d['t_observer'])
print("Moving Position", formatResult(res_1d['d_moving']))
print("Moving Time", res_1d['t_moving'])


d = SPEED_OF_LIGHT * torch.tensor([[8.0, 0.0, 0.0]])
t = torch.tensor([10.0])
u = 0.8 * SPEED_OF_LIGHT * torch.tensor([[1, 0, 0]]) # move this to vector decomposition

res = lorentz.compute(d, t, u)
print("Computation in 3D")
print("Observer Position", formatResult(res['d_observer']))
print("Observer Time", res['t_observer'])
print("Moving Position", formatResult(res['d_moving']))
print("Moving Time", res['t_moving'])
