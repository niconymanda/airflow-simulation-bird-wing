# grid size (nx, ny), 
# time step (dt), and 
# physical parameters (rho, nu). 
# velocity fields (u,v) and 
# pressure (p) to zeros or a uniform flow.
import numpy as np
import argparse
from solver import Solver
from wing import Wing

def main():
    parser = argparse.ArgumentParser(description="Simulate the incompressible Navier–Stokes equations on a 2D grid.")

    parser.add_argument("--grid-size", type=int, nargs = 2, metavar = ("NX", "NY"), default = [128, 128], help = "Grid resolution as two integers NX NY (e.g. 128 128).")
    parser.add_argument("--number-time-steps", type = int, default = 100, help = "Number of time step t for the simulation.")
    parser.add_argument("--time-step", type=float, default=0.001, help="Time step delta t for the simulation.")
    parser.add_argument("--physical-parameters", type = float, nargs = 2, metavar = ("RHO", "NU"), default = [1.0, 0.1], help = "Fluid density ρ and kinematic viscosity ν.")
    parser.add_argument("--velocity-fields", type = float, nargs = 2, metavar = ("U0", "V0"), default = [0.0, 0.0], help = "Initial uniform velocity field (u0 v0). Use 0 0 for fluid at rest.")
    parser.add_argument("--pressure", type = float, default = 0.0, help = "Initial uniform pressure p0.")
    parser.add_argument("--pad", type = float, default = 50.0, help = "Padding around the wing")
    
    args = parser.parse_args()

    u0, v0 = tuple(args.velocity_fields)
    nx, ny = tuple(args.grid_size) 
    rho, nu = tuple(args.physical_parameters) 

    u = np.full((ny, nx), u0)
    v = np.full((ny, nx), v0)
    p = np.full((ny, nx), args.pressure)

    wing = Wing("wing.stl", nx=nx, ny=ny, pad=args.pad)
    wing_mask = wing.create_mask()

    Lx, Ly = 2.0, 1.0
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)

    solver = Solver(rho, nu, u, v, p, args.number_time_steps, dx, dy, nit=50)
    
    solver.run_simulation(
        steps = args.number_time_steps,
        wing_mask = wing_mask,
        inflow = 1.0
    )

if __name__ == "__main__":
    main()