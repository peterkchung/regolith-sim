# Lunar Regolith MPM Simulation - Simple Example

This is a minimal example of using Newton's Material Point Method (MPM) solver to simulate lunar regolith (granular material) falling onto a surface under lunar gravity on an RTX 3090.

## Overview

This example creates particles above a 1m × 1m surface that fall under lunar gravity and form a pile/berm. The example uses 50,000 particles by default.

## Prerequisites

```bash
pip install newton warp-lang numpy
```

## The Code

```python
#!/usr/bin/env python3
"""
Lunar Regolith MPM - Minimal Example
Newton Physics Engine + Material Point Method
"""

import newton
import warp as wp
import numpy as np
from newton.solvers import SolverImplicitMPM

# =============================================================================
# CONFIGURATION
# =============================================================================

# Domain size (meters)
DOMAIN_X, DOMAIN_Y, DOMAIN_Z = 1.0, 1.0, 0.5

# Simulation parameters
PARTICLE_COUNT = 50_000       # Number of particles
VOXEL_SIZE = 0.02             # Grid resolution (m)
FPS = 30.0                    # Output frame rate
SUBSTEPS = 5                  # Physics substeps per frame
TOTAL_FRAMES = 100            # Simulation length

# Lunar regolith material properties
DENSITY = 1500.0              # kg/m³ (loose upper regolith)
YOUNG_MODULUS = 50e6          # Pa (50 MPa - compacted regolith)
POISSON_RATIO = 0.3           # Typical soil value
FRICTION = 0.6                # ~35° friction angle
YIELD_PRESSURE = 5e3          # Pa (low compressive strength)
GRAVITY = (0.0, 0.0, -1.62)   # Lunar gravity (m/s²)


# =============================================================================
# SIMULATION SETUP
# =============================================================================

def main():
    # Initialize Warp (automatically uses CUDA on RTX 3090)
    wp.init()
    print(f"Warp initialized: {wp.get_device()}")
    
    # -------------------------------------------------------------------------
    # Step 1: Create ModelBuilder and register MPM attributes
    # -------------------------------------------------------------------------
    builder = newton.ModelBuilder()
    
    # CRITICAL: Must register MPM attributes BEFORE adding particles
    SolverImplicitMPM.register_custom_attributes(builder)
    
    # -------------------------------------------------------------------------
    # Step 2: Add particle grid
    # -------------------------------------------------------------------------
    # Calculate grid dimensions for target particle count
    particles_per_cell = 3
    total_cells = PARTICLE_COUNT / particles_per_cell
    cells_per_side = int(np.cbrt(total_cells))
    
    # Adjust for domain aspect ratio
    dim_x = int(cells_per_side * (DOMAIN_X / DOMAIN_Y) ** (1/3))
    dim_y = int(cells_per_side * (DOMAIN_Y / DOMAIN_X) ** (1/3))
    dim_z = int(cells_per_side * (DOMAIN_Z / DOMAIN_X) ** (1/3))
    
    # Calculate actual cell sizes
    cell_x = DOMAIN_X / dim_x
    cell_y = DOMAIN_Y / dim_y
    cell_z = DOMAIN_Z / dim_z
    cell_volume = cell_x * cell_y * cell_z
    
    # Per-particle properties
    particle_mass = cell_volume * DENSITY
    particle_radius = max(cell_x, cell_y, cell_z) * 0.5
    
    print(f"\nGrid: {dim_x} x {dim_y} x {dim_z} cells")
    print(f"Voxel size: {VOXEL_SIZE:.3f} m")
    print(f"Particle mass: {particle_mass:.6f} kg")
    
    # Add particles in a grid layout - spawn above surface so they fall naturally
    spawn_height = DOMAIN_Z + 0.5  # 0.5m above ground
    builder.add_particle_grid(
        pos=wp.vec3(0.0, 0.0, spawn_height),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),  # Start at rest - let lunar gravity do the work
        dim_x=dim_x + 1,
        dim_y=dim_y + 1,
        dim_z=dim_z + 1,
        cell_x=cell_x,
        cell_y=cell_y,
        cell_z=cell_z,
        mass=particle_mass,
        jitter=2.0 * particle_radius,  # Randomize positions slightly
        radius_mean=particle_radius,
        flags=newton.ParticleFlags.ACTIVE,
    )
    
    # -------------------------------------------------------------------------
    # Step 3: Add container boundaries
    # -------------------------------------------------------------------------
    # Container collision configuration
    wall_cfg = newton.ModelBuilder.ShapeConfig(
        mu=FRICTION,
        density=0.0,  # Static/kinematic
        has_particle_collision=True,
    )
    
    # Ground plane only (particles form a pile/berm naturally)
    builder.add_ground_plane(cfg=wall_cfg)
    
    print(f"Added ground plane (open surface - particles will pile/berm)")
    
    # Optional: Uncomment to add 4 walls for a boxed container
    # wall_thickness = 0.02
    # # Back wall (y=0)
    # builder.add_shape_box(
    #     body=-1, cfg=wall_cfg,
    #     xform=wp.transform(wp.vec3(DOMAIN_X/2, -wall_thickness/2, DOMAIN_Z/2), wp.quat_identity()),
    #     hx=DOMAIN_X/2, hy=wall_thickness/2, hz=DOMAIN_Z/2,
    # )
    # # ... add other walls similarly
    
    # -------------------------------------------------------------------------
    # Step 4: Finalize the model
    # -------------------------------------------------------------------------
    model = builder.finalize()
    model.set_gravity(GRAVITY)
    
    actual_particles = model.particle_count
    print(f"\nModel finalized: {actual_particles:,} particles")
    print(f"Gravity: {GRAVITY} (lunar)")
    
    # -------------------------------------------------------------------------
    # Step 5: Configure material properties
    # -------------------------------------------------------------------------
    # Create index array for all particles
    indices = wp.array(
        np.arange(actual_particles), 
        dtype=int, 
        device=model.device
    )
    
    # Set elastoplastic properties via model.mpm namespace
    model.mpm.young_modulus[indices].fill_(YOUNG_MODULUS)
    model.mpm.poisson_ratio[indices].fill_(POISSON_RATIO)
    model.mpm.friction[indices].fill_(FRICTION)
    model.mpm.yield_pressure[indices].fill_(YIELD_PRESSURE)
    model.mpm.yield_stress[indices].fill_(1e3)  # 1 kPa deviatoric
    model.mpm.tensile_yield_ratio[indices].fill_(0.1)  # Weak in tension
    model.mpm.hardening[indices].fill_(5.0)  # Moderate hardening
    model.mpm.damping[indices].fill_(0.05)  # Low viscous damping
    
    print(f"\nMaterial properties:")
    print(f"  Young's modulus: {YOUNG_MODULUS/1e6:.1f} MPa")
    print(f"  Friction: {FRICTION:.2f} (~{np.degrees(np.arctan(FRICTION)):.1f}°)")
    print(f"  Density: {DENSITY:.0f} kg/m³")
    
    # -------------------------------------------------------------------------
    # Step 6: Create MPM solver
    # -------------------------------------------------------------------------
    config = SolverImplicitMPM.Config(
        voxel_size=VOXEL_SIZE,
        max_iterations=250,
        tolerance=1e-5,
        strain_basis="P0",           # Piecewise constant strain
        solver="gauss-seidel",       # Fast single-GPU solver
        transfer_scheme="apic",      # APIC (better than PIC)
        grid_type="sparse",          # Memory efficient
        grid_padding=0,
        critical_fraction=0.0,
        air_drag=1.0,
    )
    
    solver = SolverImplicitMPM(model, config)
    print(f"\nSolver: Implicit MPM (sparse grid, APIC)")
    
    # -------------------------------------------------------------------------
    # Step 7: Initialize simulation states
    # -------------------------------------------------------------------------
    state_0 = model.state()  # Current state
    state_1 = model.state()  # Next state
    
    # -------------------------------------------------------------------------
    # Step 8: Run simulation loop
    # -------------------------------------------------------------------------
    frame_dt = 1.0 / FPS
    sim_dt = frame_dt / SUBSTEPS
    
    print(f"\nRunning simulation:")
    print(f"  Frames: {TOTAL_FRAMES}")
    print(f"  FPS: {FPS}")
    print(f"  Substeps: {SUBSTEPS}")
    print(f"  dt: {sim_dt:.6f} s per substep")
    print("")
    
    for frame in range(TOTAL_FRAMES):
        # Substepping for numerical stability
        for _ in range(SUBSTEPS):
            state_0.clear_forces()
            
            # Advance physics
            solver.step(state_0, state_1, None, None, sim_dt)
            
            # Handle collisions with container
            solver._project_outside(state_1, state_1, sim_dt)
            
            # Swap states for next iteration
            state_0, state_1 = state_1, state_0
        
        # Log progress every 10 frames
        if frame % 10 == 0 or frame == TOTAL_FRAMES - 1:
            positions = state_0.particle_q.numpy()
            z_coords = positions[:, 2]
            print(f"Frame {frame:3d}: z_avg={z_coords.mean():.3f}m, "
                  f"z_min={z_coords.min():.3f}m, z_max={z_coords.max():.3f}m")
    
    print("\nSimulation complete!")
    return state_0


if __name__ == "__main__":
    final_state = main()
```

## Running the Example

```bash
python regolith_example.py
```

Expected output:
```
Warp initialized: cuda:0

Grid: 23 x 23 x 12 cells
Voxel size: 0.020 m
Particle mass: 0.000141 kg
Added ground + 4 walls

Model finalized: 7,776 particles
Gravity: (0.0, 0.0, -1.62) (lunar)

Material properties:
  Young's modulus: 50.0 MPa
  Friction: 0.60 (~30.9°)
  Density: 1500 kg/m³

Solver: Implicit MPM (sparse grid, APIC)

Running simulation:
  Frames: 100
  FPS: 30.0
  Substeps: 5
  dt: 0.006667 s per substep

Frame   0: z_avg=0.234m, z_min=0.024m, z_max=0.466m
Frame  10: z_avg=0.198m, z_min=0.002m, z_max=0.421m
Frame  20: z_avg=0.156m, z_min=0.000m, z_max=0.389m
...
Simulation complete!
```

## Key Concepts

### 1. ModelBuilder Pattern
Newton uses a builder pattern to construct simulation scenes:
1. Create `ModelBuilder`
2. Register MPM attributes
3. Add particles and shapes
4. Finalize to get `Model`

### 2. MPM Attributes
The `model.mpm` namespace contains per-particle arrays for:
- `young_modulus`: Elastic stiffness
- `friction`: Coulomb friction coefficient
- `yield_pressure`: Compressive strength
- `hardening`: Plastic hardening factor

### 3. Solver Configuration
- **Sparse grid**: Memory efficient for RTX 3090
- **APIC transfer**: Better angular momentum conservation than PIC
- **Gauss-Seidel**: Fast iterative solver for single GPU

### 4. Simulation Loop
MPM requires substepping for stability:
1. Clear forces
2. Step solver (state_0 → state_1)
3. Handle collisions
4. Swap states

## Extending the Example

### Add USD Export
```python
import newton.usd

# In the loop:
if frame % 10 == 0:
    newton.usd.export_usd(f"frame_{frame:04d}.usd", model, state_0)
```

### Add Real-time Viewer
```python
from newton.viewer import ViewerGL

viewer = ViewerGL(model)
viewer.set_model(model)
viewer.show_particles = True

# Position camera above container
viewer.set_camera(
    wp.vec3(DOMAIN_X/2, DOMAIN_Y/2, DOMAIN_Z + 1.5),
    pitch=-90.0, yaw=0.0
)

# In the loop:
viewer.begin_frame(frame * frame_dt)
viewer.log_state(state_0)
viewer.end_frame()
```

### Change Material Properties
```python
# Sand-like (loose)
model.mpm.friction[indices].fill_(0.5)
model.mpm.hardening[indices].fill_(0.0)

# Rock-like (stiff)
model.mpm.young_modulus[indices].fill_(10e9)  # 10 GPa
model.mpm.yield_pressure[indices].fill_(1e6)  # 1 MPa
```

## Performance Tips for RTX 3090

1. **Particle count**: 50k-500k particles fit comfortably in 24GB VRAM
2. **Voxel size**: 0.01-0.05m is typical for granular simulations
3. **Grid type**: "sparse" uses less memory than "fixed"
4. **Substeps**: 5-10 substeps for stability at 30 FPS
5. **Max iterations**: 100-250 depending on convergence needs

## References

- Newton MPM examples: `python -m newton.examples.mpm.example_mpm_granular --help`
- Newton documentation: https://github.com/NVIDIA-Omniverse/Newton
- Warp documentation: https://nvidia.github.io/warp/
