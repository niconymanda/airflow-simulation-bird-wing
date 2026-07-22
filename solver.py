import numpy as np
import matplotlib.pyplot as plt

# The core loop repeats until the solution reaches a steady state. The steps are:
#   1. Build the Source Term (b): Calculate the divergence of the velocity terms.
#   2. Solve Pressure: Solve the Poisson equation for p using Jacobi iteration.
#   3. Update Velocity: Use the new pressure to update u and v.
#   4. Apply Boundary Conditions: Enforce inlet velocity and zero velocities inside the wing.

class Solver:
    def __init__(self, rho, nu, u, v, p, dt, dx, dy, nit = 50):
        self.rho = rho
        self.nu = nu
        self.u = u
        self.v = v
        self.p = p
        self.dt = dt
        self.dx = dx
        self.dy = dy
        self.b = np.zeros_like(p)
        self.nit = nit

    def build_source_term(self): 
        """
        Build the Source Term (b):
            The source term we need to find the thing that tells us “how much pressure curvature do we need, and where?”
            It essentially measures how much the current velocity field "wants" to locally compress/expand and how strong its local shear is.
            The pressure solve then finds a pressure field whose curvature counteracts that, so that after the pressure correction 
            the updated velocity field is (as close as possible to) divergence-free.
        """
        self.b[1:-1, 1:-1] = (
            self.rho * (
                1 / self.dt * (
                    (self.u[1:-1, 2:] - self.u[1:-1, 0:-2]) / (2 * self.dx) +
                    (self.v[2:, 1:-1] - self.v[0:-2, 1:-1]) / (2 * self.dy)
                ) - (
                    (self.u[1:-1, 2:] - self.u[1:-1, 0:-2]) / (2 * self.dx)
                )**2 - 2 * (
                    (self.u[2:, 1:-1] - self.u[0:-2, 1:-1]) / (2 * self.dy) *
                    (self.v[1:-1, 2:] - self.v[1:-1, 0:-2]) / (2 * self.dx)
                ) - (
                    (self.v[2:, 1:-1] - self.v[0:-2, 1:-1]) / (2 * self.dy)
                )**2
            )
        )
 
    def solve_pressure(self):
        """
        We "spread out" the pressure information across the grid until it becomes consistent with the source term.
        This is the step that produces the pressure field needed to enforce incompressibility.

        NB : The Jacobi iteration repeatedly replaces each interior cell's pressure with an average of its neighbours, nudged by b. 
        Boundary values are pinned here to set a reference pressure.
        """
        for _ in range(self.nit):
            pn = self.p.copy()
            self.p[1:-1, 1:-1] = (
                    (pn[1:-1, 2:] + pn[1:-1, 0:-2]) * self.dy**2 +
                    (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * self.dx**2 -
                    self.b[1:-1, 1:-1] * self.dx**2 * self.dy**2
                ) / (
                    2 * (self.dx**2 + self.dy**2)
                )
            
            # Boundary conditions: p=0 at edges (gauge pressure)
            self.p[:, -1] = 0
            self.p[:, 0]  = 0
            self.p[-1, :] = 0
            self.p[0, :]  = 0
    
    def update_velocity(self):
        """
        Advances the velocity field by one time step using the momentum equation (discretised).

        Intuition: take the current velocity and then:
            -> move it along with the flow (advection: "the flow carries itself"),
            -> smooth it out (viscosity/diffusion: internal friction),
            -> and finally push/pull it using pressure differences (pressure gradient), which is the mechanism that will make the 
               updated field compatible with incompressibility after the pressure solve.

        Practically: this computes new u and v on interior cells from the old values plus neighbour differences (finite differences).
        """

        un = self.u.copy()
        vn = self.v.copy() 
        # Update u (x-velocity)
        self.u[1:-1, 1:-1] = (
            un[1:-1, 1:-1] - un[1:-1, 1:-1] * self.dt / self.dx * (
                un[1:-1, 1:-1] - un[1:-1, 0:-2]
            ) - vn[1:-1, 1:-1] * self.dt / self.dy * (
                un[1:-1, 1:-1] - un[0:-2, 1:-1]
                ) - self.dt / (
                    2 * self.rho * self.dx
                ) * (
                    self.p[1:-1, 2:] - self.p[1:-1, 0:-2]
                ) + self.nu * (
                    self.dt / self.dx**2 * (
                        un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]
                    ) + self.dt / self.dy**2 * (
                        un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1]
                    )
                )
            )
        # Update v (y-velocity)
        self.v[1:-1, 1:-1] = (
            vn[1:-1, 1:-1] - un[1:-1, 1:-1] * self.dt / self.dx * (
                vn[1:-1, 1:-1] - vn[1:-1, 0:-2]
            ) - vn[1:-1, 1:-1] * self.dt / self.dy * (
                vn[1:-1, 1:-1] - vn[0:-2, 1:-1]
            ) - self.dt / (
                2 * self.rho * self.dy
            ) * (
                self.p[2:, 1:-1] - self.p[0:-2, 1:-1]
            ) + self.nu * (
                self.dt / self.dx**2 * (
                    vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]
                ) + self.dt / self.dy**2 * (
                    vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1]
                )
            )
        )
    
    def apply_boundary_conditions(self, wing_mask, inflow = 1.0):
        """enforces physical constraints of the problem after each update"""
        self.u[wing_mask] = 0
        self.v[wing_mask] = 0

        # Moving horizontally
        self.u[:, 0] = inflow 
        self.u[:, -1] = self.u[:, -2] 
        self.v[:, 0] = 0
        self.v[:, -1] = self.v[:, -2] 

        # Top and bottom
        self.u[0, :] = self.u[1, :]
        self.u[-1, :] = self.u[-2, :] 
        self.v[0, :] = 0
        self.v[-1, :] = 0

    def step(self, wing_mask, inflow=1.0):
        """Performs one timestep in the simulation"""
        self.build_source_term()
        self.solve_pressure()
        self.update_velocity()
        self.apply_boundary_conditions(wing_mask, inflow=inflow)

    def run_simulation(self, steps, wing_mask, inflow=1.0, plot_every=10):
        """Run the solver and visualize the flow as it develops over time.

        Shows:
        - background color: speed magnitude
        - arrows: velocity field
        - black region: wing/solid mask
        """
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 5))

        for n in range(steps):
            self.step(wing_mask, inflow=inflow)

            if n % plot_every == 0:
                ax.clear()

                speed = np.sqrt(self.u**2 + self.v**2)

                im = ax.imshow(
                    speed,
                    origin="lower",
                    cmap="coolwarm",
                    aspect="auto"
                )

                skip = 4
                y, x = np.mgrid[0:self.u.shape[0], 0:self.u.shape[1]]
                ax.quiver(
                    x[::skip, ::skip],
                    y[::skip, ::skip],
                    self.u[::skip, ::skip],
                    self.v[::skip, ::skip],
                    color="white",
                    scale=25
                )

                if wing_mask is not None:
                    ax.contour(
                        wing_mask.astype(float),
                        levels=[0.5],
                        colors="black",
                        linewidths=2
                    )
                    ax.imshow(
                        np.ma.masked_where(~wing_mask, wing_mask),
                        origin="lower",
                        cmap="gray",
                        alpha=0.9
                    )

                ax.set_title(f"Velocity field at step {n}")
                ax.set_xlabel("x")
                ax.set_ylabel("y")

                plt.pause(0.01)

        plt.ioff()
        plt.show()
