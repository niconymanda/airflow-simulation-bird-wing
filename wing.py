import numpy as np
from stl import mesh
from matplotlib.path import Path
import matplotlib.pyplot as plt


class Wing:
    def __init__(self, mesh_file: str, nx: int, ny: int, pad: float):
        """
        mesh_file: path to STL file with an extruded airfoil (e.g. wing.stl)
        nx, ny: number of grid points in x and y
        pad: padding added around the wing in physical units
        """
        self.mesh = mesh.Mesh.from_file(mesh_file)
        self.nx = nx
        self.ny = ny
        self.pad = pad

    def _domain_extents(self):
        """
        Compute padded x/y extents based on the STL bounding box.
        """
        x_min, y_min, _ = self.mesh.min_
        x_max, y_max, _ = self.mesh.max_

        x_min -= self.pad
        x_max += self.pad
        y_min -= self.pad
        y_max += self.pad

        return x_min, x_max, y_min, y_max

    def _airfoil_polygon(self) -> np.ndarray:
        """
        Extract a 2D airfoil polygon from the 3D STL by:
        - projecting vertices to (x, y),
        - removing duplicates,
        - ordering them by angle around the centroid.
        Returns an array of shape (N, 2).
        """
        # (N_tri, 3, 3) -> (N_pts, 3)
        verts = self.mesh.vectors.reshape(-1, 3)
        xy = verts[:, :2]

        # Remove duplicate points
        xy = np.unique(xy, axis=0)

        # Order points roughly along the boundary by angle
        center = xy.mean(axis=0)
        angles = np.arctan2(xy[:, 1] - center[1], xy[:, 0] - center[0])
        order = np.argsort(angles)
        poly = xy[order]

        return poly

    def create_mask(self) -> np.ndarray:
        """
        Create a Boolean mask of shape (ny, nx) where True = inside wing.
        """
        x_min, x_max, y_min, y_max = self._domain_extents()

        xs = np.linspace(x_min, x_max, self.nx)
        ys = np.linspace(y_min, y_max, self.ny)
        X, Y = np.meshgrid(xs, ys)  # X[j,i], Y[j,i]

        poly = self._airfoil_polygon()
        wing_path = Path(poly)

        points = np.vstack([X.ravel(), Y.ravel()]).T  # (ny*nx, 2)
        mask_1d = wing_path.contains_points(points)
        wing_mask = mask_1d.reshape(self.ny, self.nx)

        return wing_mask

    def visualise_mask(self, filename: str | None = None):
        """
        Visualise the 2D wing mask.
        If filename is given, save the plot to that path; otherwise show it.
        """
        mask = self.create_mask()
        x_min, x_max, y_min, y_max = self._domain_extents()

        plt.figure(figsize=(6, 4))
        plt.imshow(
            mask,
            origin="lower",
            extent=[x_min, x_max, y_min, y_max],
            cmap="Greys_r",
            interpolation="nearest",
        )
        plt.colorbar(label="Inside wing (True=1, False=0)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Wing mask on simulation grid")

        if filename is not None:
            plt.savefig(filename, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


if __name__ == "__main__":
    NX = 128
    NY = 64
    PAD = 50.0  # adjust to control padding around the wing

    wing = Wing("wing.stl", nx=NX, ny=NY, pad=PAD)
    wing.visualise_mask("wing_mask.png")