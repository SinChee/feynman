import torch
import torch.nn as nn

from utils import *


class LorentzRelativity(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def compute_moving(self, 
                       distance: torch.Tensor, 
                       time: torch.Tensor, 
                       velocity: torch.Tensor):

        """Compute the distance and time of moving spaceship from observer
        TODO: frame a problem statement here

        Args:
            distance (torch.Tensor): distance of observer
            time (torch.Tensor): time of observer
            velocity (torch.Tensor): speed of moving spaceship (relative to observer)
            speed_ratio (int, optional): _description_. Defaults to 1.

        Returns:
            res (Dict): Results containing observer and moving
        """

        _beta = speedRatio(velocity=velocity)
        _gamma = self.lorentz_factor(velocity=velocity)
        distance_moving = _gamma * (distance - _beta * SPEED_OF_LIGHT * time)
        spacetime_moving = _gamma * (SPEED_OF_LIGHT * time - _beta * distance)
        res = {
            "d_observer": distance,
            "t_observer": time,
            "d_moving": distance_moving,
            "t_moving": spacetime_moving / SPEED_OF_LIGHT,
        }
        return res

    def compute(self, position: torch.Tensor, time: torch.Tensor, velocity: torch.Tensor):
        """Compute the lorentz transform in 3D space. 
        Refer to https://physics.stackexchange.com/questions/556913/general-form-of-the-lorentz-transformation

        Args:
            position (torch.Tensor): _description_
            time (torch.Tensor): _description_
            velocity (torch.Tensor): _description_
        """
        _beta = speedRatio(velocity=velocity).T             # [3, 1]
        _beta2 = torch.matmul(_beta.T, _beta)               # [1, 1]: scaler
        _gamma = self.lorentz_factor(velocity=velocity).T   # [1, 1]: scaler
        
        spacetime = (SPEED_OF_LIGHT * time).unsqueeze(0)    # space time of observer
        x_o = torch.concat([spacetime, position.T])         # position of observer

        # build lorentz matrix
        lorentz_matrix = torch.ones(4, 4, device=spacetime.device) * _gamma
        lorentz_matrix[1, 1:] = (_gamma - 1) * _beta[0] / _beta2 * _beta.T
        lorentz_matrix[2, 1:] = (_gamma - 1) * _beta[1] / _beta2 * _beta.T
        lorentz_matrix[3, 1:] = (_gamma - 1) * _beta[2] / _beta2 * _beta.T
        lorentz_matrix[0, 1:] = (-_gamma*_beta).squeeze(1)
        lorentz_matrix[1:, 0] = (-_gamma*_beta).squeeze(1)
        lorentz_matrix[1:, 1:] += torch.eye(3, 3)

        x_m = torch.matmul(lorentz_matrix, x_o)

        res = {
            "d_observer": position,
            "t_observer": time,
            "d_moving": x_m.T[0, 1:],
            "t_moving": x_m[0] / SPEED_OF_LIGHT,
        }
        return res


    def lorentz_factor(self, velocity: torch.Tensor) -> torch.Tensor:
        """Compute the lorentz factor
        Known bug: still return value when travelling in speed of light

        Args:
            velocity (torch.Tensor): velocity of moving object (relativity)

        Returns:
            torch.Tensor: gamma
        """
        beta = speedRatio(velocity=velocity)
        gamma = 1.0 / torch.sqrt(1- torch.matmul(beta, beta.T))
        return gamma



