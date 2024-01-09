
import torch
import numpy as np
from kornia import create_meshgrid
from tqdm import tqdm

class PositionalEmbedding(torch.nn.Module):
    def __init__(self, max_length: float, nb_levels: int) -> None:
        super(PositionalEmbedding, self).__init__()
        self.max_length = max_length
        self.nb_levels = nb_levels
        
    def forward(self, X):
        outputs = []
        period = self.max_length

        for level in range(self.nb_levels):
            outputs.append(torch.sin(2*torch.pi * X/period))
            outputs.append(torch.cos(2*torch.pi * X/period))
            period /= 2
        outputs = torch.concatenate(outputs, dim=1)

        return outputs

class RnerfModel(torch.nn.Module):
    def __init__(self, hidden_layers: list[int], nb_positional_levels = 10, nb_directional_levels = 10) -> None:
        super(RnerfModel, self).__init__()
        assert len(hidden_layers) > 0

        self.hidden_layers = hidden_layers

        self.nb_positional_levels = nb_positional_levels
        self.nb_directional_levels = nb_directional_levels
        

        self.positional_embedding = PositionalEmbedding(10,nb_positional_levels)
        self.direction_embedding = PositionalEmbedding(2,nb_directional_levels)

        nb_positional_embeding = 2 * 3 * nb_positional_levels
        nb_directional_embeding = 2 * 3 * nb_directional_levels
        self.main = torch.nn.Sequential(
            torch.nn.Linear(nb_positional_embeding + nb_directional_embeding, hidden_layers[0]),
            torch.nn.ReLU(),
        )

        for layer_index in range(1, len(hidden_layers)):
            self.main.add_module("hidden_layer_"+str(layer_index), torch.nn.Linear(hidden_layers[layer_index-1], hidden_layers[layer_index]))
            self.main.add_module("activation_layer_"+str(layer_index), torch.nn.ReLU())

        self.main.add_module(
            "output_layer",
            torch.nn.Linear(hidden_layers[-1], 3)
        )
        self.main.add_module("output_activation", torch.nn.Sigmoid())

        self.losses = []
    
    def forward(self, X):
        embeded_positions = self.positional_embedding(X[..., :3])
        embeded_directions = self.positional_embedding(X[..., 3:])
        return self.main(torch.concatenate([embeded_positions, embeded_directions], dim=1))
    
    def train(self, rays, rgbs, nb_epochs: int, batch_size: int, lr=1e-3):

        losses = []
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(nb_epochs):
            batch_indices = torch.split(torch.randperm(len(rays)), batch_size)

            for batch_index in tqdm(range(len(batch_indices))):
                ray_batch = rays[batch_indices[batch_index]]
                rgb_batch = rgbs[batch_indices[batch_index]]

                optimizer.zero_grad()

                outputs = self.forward(ray_batch)

                loss = torch.nn.functional.mse_loss(outputs, rgb_batch)
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                self.losses.append(loss.item())
        return losses
    
    def save(self):
        torch.save(self, "./results/models/rnerf_layers_"+str(self.hidden_layers)+"_nbposlvl_"+str(self.nb_positional_levels)+"_nbdirlvl_"+str(self.nb_directional_levels))
        


class RaysRenderer:
    def __init__(self, model):
        self.model = model

    def render(self, img_wh: tuple[int, int], origin: torch.Tensor, global_direction:torch.Tensor, fov: float):
        w, h = img_wh
        focal = 0.5*800/np.tan(0.5*fov) # original focal length
                                                                    # when W=800
        focal *= w/800 # modify focal length to match size img_wh

        rays_direction = get_ray_directions(h,w, focal)
        rays_direction += global_direction
        
        inputs = torch.reshape(rays_direction, (w*h, 3))
        inputs = torch.concatenate([inputs, origin[None, ...].repeat(inputs.shape[0], 1)],dim=1)

        outputs = self.model(inputs)

        img = torch.reshape(outputs.detach(), (w,h,3))

        return img
    
    def render_rays(self, img_wh: tuple[int, int], rays: torch.Tensor, fov: float):
        w, h = img_wh

        inputs = rays
        outputs = self.model(inputs)

        img = torch.reshape(outputs.detach(), (w,h,3))

        return img

        


def get_ray_directions(H, W, focal):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W, focal: image height, width and focal length

    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    i, j = grid.unbind(-1)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    directions = \
        torch.stack([(i-W/2)/focal, -(j-H/2)/focal, -torch.ones_like(i)], -1) # (H, W, 3)

    return directions