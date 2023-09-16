import torch
import torch.nn as nn
import matplotlib.pyplot as plt

### Constants
SPEED_OF_LIGHT = 299792458.0 # m/s


def speedRatio(velocity: torch.Tensor) -> torch.Tensor:
    """Compute the ratio of velocity to speed of light

    Args:
        velocity (torch.Tensor): velocity of moving object (relativity)

    Returns:
        torch.Tensor: ratio of velocity to speed of light (beta)
    """
    return velocity / SPEED_OF_LIGHT

def gamma(velocity: torch.Tensor) -> torch.Tensor:
    """Compute gamma

    Args:
        velocity (torch.Tensor): velocity of moving object (relativity)

    Returns:
        torch.Tensor: gamma
    """
    beta = speedRatio(velocity=velocity)
    gamma = 1.0 / torch.sqrt(1- torch.matmul(beta, beta))
    return gamma

def formatResult(res):
    return res / SPEED_OF_LIGHT
    


# Some constants that is common in physics
# class Convertion(nn.Module):
#     def __init__(self, device="cuda") -> None:
#         super().__init__()
#         # should store constants in the form of dictionary for metadata storage such as description, unit, etc
#         # when called, only return the tensor, something like this
#         # self.C = {
#         #     'value': torch.Tensor(299792458),
#         #     'unit': 'm/s',
#         #     'decription': 'constant speed of light in vaccum in m/s'
#         # }
#         self.C = torch.Tensor(299792458, dtype=torch.float32, device=device)

    

