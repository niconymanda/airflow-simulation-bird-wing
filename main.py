# grid size (nx, ny), 
# time step (dt), and 
# physical parameters (rho, nu). 
# velocity fields (u,v) and 
# pressure (p) to zeros or a uniform flow.

import argparse

def main():
    parser = argparse.ArgumentParser(description="Simulate the incompressible Navier–Stokes equations on a 2D grid.")

    parser.add_argument("--grid-size", type=int, nargs = 2, metavar = ("NX", "NY"), default = [128, 128], help = "Grid resolution as two integers NX NY (e.g. 128 128).")
    parser.add_argument("--time-step", type = float, default = 0.001, help = "Time step Δt for the simulation (in seconds).")
    parser.add_argument("--physical-parameters", type = float, nargs = 2, metavar = ("RHO", "NU"), default = [1.0, 0.1], help = "Fluid density ρ and kinematic viscosity ν.")
    parser.add_argument("--velocity-fields", type = float, nargs = 2, metavar = ("U0", "V0"), default = [0.0, 0.0], help = "Initial uniform velocity field (u0 v0). Use 0 0 for fluid at rest.")
    parser.add_argument("--pressure", type = float, default = 0.0, help = "Initial uniform pressure p0.")
    
    args = parser.parse_args()
    
    run_simulation(
        grid_size = tuple(args.grid_size),
        time_step = args.time_step,
        physical_parameters = tuple(args.physical_parameters),
        velocity_fields = tuple(args.velocity_fields),
        pressure = args.pressure,
    )

if __name__ == "__main__":
    main()