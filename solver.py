import numpy as np
# The core loop repeats until the solution reaches a steady state. The steps are:
#   1. Build the Source Term (b): Calculate the divergence of the velocity terms.
#   2. Solve Pressure: Solve the Poisson equation for p using Jacobi iteration.
#   3. Update Velocity: Use the new pressure to update u and v.
#   4. Apply Boundary Conditions: Enforce inlet velocity and zero velocities inside the wing.

class Solver:
    def __init__(self, rho, u, v, p, dt, dx, dy, nit = 50):
        self.rho = rho
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
            b = (ρ/Δt)⋅∇⋅v* , 
            where by v* is tentative velocity you got after advection + diffusion (before applying pressure correction)
            Essentially when ∇⋅v* is nonzero somewhere, it tells us, there is “compression” or “expansion” somewhere. 
            If ∇⋅v* > 0 => b > 0 and if ∇⋅v* < 0 => b < 0
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
        """Solve Pressure"""
        for _ in range(self.nit):
            pn = self.p.copy()
            self.p[1:-1, 1:-1] = (
                (pn[1:-1, 2:] + pn[1:-1, 0:-2]) * self.dy**2 +
                (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * self.dx**2 -
                self.b[1:-1, 1:-1] * self.dx**2 * self.dy**2
            ) / (2 * (self.dx**2 + self.dy**2))
            
            # Boundary conditions: p=0 at edges (gauge pressure)
            self.p[:, -1] = 0
            self.p[:, 0]  = 0
            self.p[-1, :] = 0
            self.p[0, :]  = 0
    
    def update_velocity(self):
        """Update Velocity"""
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