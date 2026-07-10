import numpy as np
from stl import mesh
from matplotlib.path import Path
# 'wing.stl'
class Wing:
    def __init__(self, mesh_file, nx, ny, pad):
        self.mesh = mesh.Mesh.from_file(mesh_file)
        self.nx = nx
        self.ny = ny
        self.pad = pad
    
    def create_mask(self):
        # TODO: make flat_vectors more robust 
        flat_vectors = self.mesh.vectors.reshape(-1, 3)
        flat_vectors = flat_vectors[:, 0:2]

        x_min, y_min, _ = self.mesh.min_
        x_max, y_max, _ = self.mesh.max_

        x_min -= self.pad
        x_max += self.pad

        y_min -= self.pad
        y_max += self.pad

        xs = np.linspace(x_min, x_max, self.nx)
        ys = np.linspace(y_min, y_max, self.ny)
        X, Y = np.meshgrid(xs, ys)


        wing_path = Path(flat_vectors)
        points = np.vstack([X.ravel(), Y.ravel()]).T
        wing_mask = wing_path.contains_points(points)
        wing_mask = wing_mask.reshape(self.ny, self.nx)

        return wing_mask


# During the simulation, we force velocity to zero at these masked points (no-slip/no-penetration condition).