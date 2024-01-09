import torch
import json
import numpy as np
import os
from PIL import Image
from torchvision import transforms

import rnerf

img_transform = transforms.ToTensor()

class RayDataset:
    def __init__(self, dataset_path: str, img_wh: tuple[int, int]) -> None:
        self.dataset_path = dataset_path
        self.img_wh = img_wh
    
    def compute_dataset(self):
        with open(os.path.join(self.dataset_path,
                                f"transforms_train.json"), 'r') as f:
            meta = json.load(f)

        w, h = self.img_wh
        focal = 0.5*800/np.tan(0.5*meta['camera_angle_x']) # original focal length
                                                                        # when W=800
        focal *= w/800 # modify focal length to match size img_wh

        # bounds, common for all scenes
        near = 2.0
        far = 6.0
        bounds = np.array([near, far])
        
        # ray directions for all pixels, same for all images (same H, W, focal)
        directions = \
            rnerf.get_ray_directions(h, w, focal) # (h, w, 3)
        
        image_paths = []
        poses = []
        rays = []
        rgbs = []
        for frame in meta['frames']:
            pose = np.array(frame['transform_matrix'])[:3, :4]
            poses += [pose]
            c2w = torch.FloatTensor(pose)

            image_path = os.path.join(self.dataset_path, f"{frame['file_path']}.png")
            image_paths += [image_path]
            img = Image.open(image_path)
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = img_transform(img) # (4, h, w)
            img = img.view(4, -1).permute(1, 0) # (h*w, 4) RGBA
            img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB
            rgbs += [img]
            
            rays_o, rays_d = self.get_rays(directions, c2w) # both (h*w, 3)

            rays += [torch.cat([rays_o, rays_d, 
                                            near*torch.ones_like(rays_o[:, :1]),
                                            far*torch.ones_like(rays_o[:, :1])],
                                            1)] # (h*w, 8)

        rays = torch.cat(rays, 0) # (len(meta['frames])*h*w, 8)
        rgbs = torch.cat(rgbs, 0) # (len(meta['frames])*h*w, 3)

        # saving
        np.save(os.path.expanduser(self.dataset_path+"computed_rays"), rays)
        np.save(os.path.expanduser(self.dataset_path+"computed_rgbs"), rgbs)

        return (rays, rgbs)
        


    def get_rays(self, directions, c2w):
        """
        Get ray origin and normalized directions in world coordinate for all pixels in one image.
        Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
                ray-tracing-generating-camera-rays/standard-coordinate-systems

        Inputs:
            directions: (H, W, 3) precomputed ray directions in camera coordinate
            c2w: (3, 4) transformation matrix from camera coordinate to world coordinate

        Outputs:
            rays_o: (H*W, 3), the origin of the rays in world coordinate
            rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
        """
        # Rotate ray directions from camera coordinate to the world coordinate
        rays_d = directions @ c2w[:, :3].T # (H, W, 3)
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        # The origin of all rays is the camera origin in world coordinate
        rays_o = c2w[:, 3].expand(rays_d.shape) # (H, W, 3)

        rays_d = rays_d.view(-1, 3)
        rays_o = rays_o.view(-1, 3)

        return rays_o, rays_d


    def get_ndc_rays(self, H, W, focal, near, rays_o, rays_d):
        """
        Transform rays from world coordinate to NDC.
        NDC: Space such that the canvas is a cube with sides [-1, 1] in each axis.
        For detailed derivation, please see:
        http://www.songho.ca/opengl/gl_projectionmatrix.html
        https://github.com/bmild/nerf/files/4451808/ndc_derivation.pdf

        In practice, use NDC "if and only if" the scene is unbounded (has a large depth).
        See https://github.com/bmild/nerf/issues/18

        Inputs:
            H, W, focal: image height, width and focal length
            near: (N_rays) or float, the depths of the near plane
            rays_o: (N_rays, 3), the origin of the rays in world coordinate
            rays_d: (N_rays, 3), the direction of the rays in world coordinate

        Outputs:
            rays_o: (N_rays, 3), the origin of the rays in NDC
            rays_d: (N_rays, 3), the direction of the rays in NDC
        """
        # Shift ray origins to near plane
        t = -(near + rays_o[...,2]) / rays_d[...,2]
        rays_o = rays_o + t[...,None] * rays_d

        # Store some intermediate homogeneous results
        ox_oz = rays_o[...,0] / rays_o[...,2]
        oy_oz = rays_o[...,1] / rays_o[...,2]
        
        # Projection
        o0 = -1./(W/(2.*focal)) * ox_oz
        o1 = -1./(H/(2.*focal)) * oy_oz
        o2 = 1. + 2. * near / rays_o[...,2]

        d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - ox_oz)
        d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - oy_oz)
        d2 = 1 - o2
        
        rays_o = torch.stack([o0, o1, o2], -1) # (B, 3)
        rays_d = torch.stack([d0, d1, d2], -1) # (B, 3)
        
        return rays_o, rays_d

if __name__ == "__main__":
    print("Generating dataset...")

    ray_dataset = RayDataset(os.path.expanduser("~/Documents/datasets/nerf_lego_dataset/"), (800,800))
    ray_dataset.compute_dataset()

