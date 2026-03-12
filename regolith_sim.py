#!/usr/bin/env python3
"""
Lunar Regolith MPM Simulation using Newton Physics Engine
Optimized for RTX 3090 with USD export and real-time viewer
"""

import newton
import newton.examples
import warp as wp
import numpy as np
import os
import argparse

from newton.solvers import SolverImplicitMPM


class LunarRegolithSimulation:
    """Lunar regolith MPM simulation using Newton's implicit MPM solver"""

    def __init__(self, viewer, options):
        """Initialize the simulation with viewer and options"""

        # Timing parameters
        self.fps = options.fps
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = options.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.total_frames = options.total_frames
        self.current_frame = 0

        # Export settings
        self.export_usd = options.export_usd
        self.output_dir = options.output_dir
        self.export_every_n_frames = options.export_every_n_frames
        if self.export_usd:
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"USD export enabled: {self.output_dir}/")

        # Save viewer reference
        self.viewer = viewer

        # Domain settings
        self.domain_x = options.domain_x
        self.domain_y = options.domain_y
        self.domain_z = options.domain_z
        self.target_particles = options.target_particles
        self.voxel_size = options.voxel_size

        # Create builder
        builder = newton.ModelBuilder()

        # Register MPM custom attributes (CRITICAL: before adding particles)
        SolverImplicitMPM.register_custom_attributes(builder)

        # Emit particles
        self.emit_particles(builder, options)

        # Add container boundaries
        self.add_container(builder, options)

        # Finalize model
        self.model = builder.finalize()
        self.model.set_gravity(options.gravity)

        print(f"\nModel finalized:")
        print(f"  Particles: {self.model.particle_count:,}")
        print(f"  Voxel size: {self.voxel_size:.4f} m")
        print(f"  Gravity: {options.gravity}")

        # Initialize states
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        # Configure MPM solver options
        mpm_config = SolverImplicitMPM.Config()
        mpm_config.voxel_size = self.voxel_size
        mpm_config.max_iterations = options.max_iterations
        mpm_config.tolerance = options.tolerance
        mpm_config.strain_basis = options.strain_basis
        mpm_config.solver = options.solver
        mpm_config.transfer_scheme = options.transfer_scheme
        mpm_config.grid_type = options.grid_type
        mpm_config.grid_padding = options.grid_padding
        mpm_config.critical_fraction = options.critical_fraction
        mpm_config.air_drag = options.air_drag

        # Initialize MPM solver
        self.solver = SolverImplicitMPM(self.model, mpm_config)

        # Set per-particle material properties via model.mpm
        self.set_material_properties(options)

        # Setup viewer
        if viewer:
            self.viewer.set_model(self.model)
            self.viewer.show_particles = True

            # Position camera above the container looking down
            # Container center is at (domain_x/2, domain_y/2, domain_z/2)
            # Position camera above looking down at -90 degrees pitch (straight down)
            camera_height = (
                max(self.domain_x, self.domain_y) * 1.5
            )  # 1.5x the domain size above
            camera_pos = wp.vec3(
                self.domain_x / 2,  # Center X
                self.domain_y / 2,  # Center Y
                self.domain_z + camera_height,  # Above container
            )

            # Set camera looking straight down (-90 pitch, 0 yaw)
            if hasattr(self.viewer, "set_camera"):
                self.viewer.set_camera(camera_pos, pitch=-90.0, yaw=0.0)
                print(
                    f"\nCamera positioned at ({camera_pos[0]:.2f}, {camera_pos[1]:.2f}, {camera_pos[2]:.2f}) looking down"
                )

        # Try to capture CUDA graph for faster simulation
        self.capture()

        print("\n" + "=" * 60)
        print("Simulation initialized successfully!")
        print("=" * 60)

    def emit_particles(self, builder, options):
        """Create particle grid for regolith bed"""

        density = options.density
        voxel_size = self.voxel_size
        particles_per_cell = 3

        # Calculate grid dimensions to achieve target particle count
        total_cells = self.target_particles / particles_per_cell
        cells_per_side = int(np.cbrt(total_cells))

        aspect_xy = self.domain_x / self.domain_y
        aspect_xz = self.domain_x / self.domain_z

        dim_x = int(cells_per_side * np.cbrt(aspect_xy * aspect_xz))
        dim_y = int(cells_per_side * np.cbrt(1.0 / aspect_xy))
        dim_z = int(cells_per_side * np.cbrt(1.0 / aspect_xz))

        # Recalculate voxel size based on actual grid
        actual_voxel = min(
            self.domain_x / dim_x, self.domain_y / dim_y, self.domain_z / dim_z
        )
        self.voxel_size = actual_voxel

        cell_size_x = self.domain_x / dim_x
        cell_size_y = self.domain_y / dim_y
        cell_size_z = self.domain_z / dim_z
        cell_volume = cell_size_x * cell_size_y * cell_size_z

        radius = max(cell_size_x, cell_size_y, cell_size_z) * 0.5
        mass = cell_volume * density

        print(f"\nParticle grid:")
        print(f"  Grid cells: {dim_x} x {dim_y} x {dim_z}")
        print(f"  Grid points: {dim_x + 1} x {dim_y + 1} x {dim_z + 1}")
        print(f"  Voxel size: {self.voxel_size:.4f} m")
        print(
            f"  Cell size: {cell_size_x:.4f} x {cell_size_y:.4f} x {cell_size_z:.4f} m"
        )
        print(f"  Particle mass: {mass:.6f} kg")
        print(f"  Target particles: {self.target_particles:,}")

        builder.add_particle_grid(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=dim_x + 1,
            dim_y=dim_y + 1,
            dim_z=dim_z + 1,
            cell_x=cell_size_x,
            cell_y=cell_size_y,
            cell_z=cell_size_z,
            mass=mass,
            jitter=2.0 * radius,
            radius_mean=radius,
            flags=newton.ParticleFlags.ACTIVE,
        )

        actual_particles = (dim_x + 1) * (dim_y + 1) * (dim_z + 1)
        print(f"  Actual particles: {actual_particles:,}")

    def add_container(self, builder, options):
        """Add rigid container boundaries"""

        # Container collision properties
        container_cfg = newton.ModelBuilder.ShapeConfig(
            mu=options.friction,
            density=0.0,  # Static
            has_particle_collision=True,
        )

        wall_thickness = 0.02  # 2cm

        # Ground plane
        builder.add_ground_plane(cfg=container_cfg)

        # Back wall (y=0)
        builder.add_shape_box(
            body=-1,
            cfg=container_cfg,
            xform=wp.transform(
                wp.vec3(self.domain_x / 2, -wall_thickness / 2, self.domain_z / 2),
                wp.quat_identity(),
            ),
            hx=self.domain_x / 2,
            hy=wall_thickness / 2,
            hz=self.domain_z / 2,
        )

        # Front wall (y=domain_y)
        builder.add_shape_box(
            body=-1,
            cfg=container_cfg,
            xform=wp.transform(
                wp.vec3(
                    self.domain_x / 2,
                    self.domain_y + wall_thickness / 2,
                    self.domain_z / 2,
                ),
                wp.quat_identity(),
            ),
            hx=self.domain_x / 2,
            hy=wall_thickness / 2,
            hz=self.domain_z / 2,
        )

        # Left wall (x=0)
        builder.add_shape_box(
            body=-1,
            cfg=container_cfg,
            xform=wp.transform(
                wp.vec3(-wall_thickness / 2, self.domain_y / 2, self.domain_z / 2),
                wp.quat_identity(),
            ),
            hx=wall_thickness / 2,
            hy=self.domain_y / 2,
            hz=self.domain_z / 2,
        )

        # Right wall (x=domain_x)
        builder.add_shape_box(
            body=-1,
            cfg=container_cfg,
            xform=wp.transform(
                wp.vec3(
                    self.domain_x + wall_thickness / 2,
                    self.domain_y / 2,
                    self.domain_z / 2,
                ),
                wp.quat_identity(),
            ),
            hx=wall_thickness / 2,
            hy=self.domain_y / 2,
            hz=self.domain_z / 2,
        )

        print(f"\nContainer:")
        print(f"  Ground plane + 4 walls")
        print(f"  Wall thickness: {wall_thickness * 100:.1f} cm")

    def set_material_properties(self, options):
        """Set per-particle material properties for lunar regolith"""

        num_particles = self.model.particle_count

        # Create index array for all particles
        import numpy as np

        indices = wp.array(
            np.arange(num_particles), dtype=int, device=self.model.device
        )

        # Set material properties via model.mpm namespace
        self.model.mpm.young_modulus[indices].fill_(options.young_modulus)
        self.model.mpm.poisson_ratio[indices].fill_(options.poisson_ratio)
        self.model.mpm.friction[indices].fill_(options.friction)
        self.model.mpm.yield_pressure[indices].fill_(options.yield_pressure)
        self.model.mpm.yield_stress[indices].fill_(options.yield_stress)
        self.model.mpm.tensile_yield_ratio[indices].fill_(options.tensile_yield_ratio)
        self.model.mpm.hardening[indices].fill_(options.hardening)
        self.model.mpm.damping[indices].fill_(options.damping)

        print(f"\nMaterial properties (lunar regolith):")
        print(f"  Young's modulus: {options.young_modulus / 1e6:.1f} MPa")
        print(f"  Poisson ratio: {options.poisson_ratio:.2f}")
        print(
            f"  Friction: {options.friction:.2f} (~{np.degrees(np.arctan(options.friction)):.1f}°)"
        )
        print(f"  Yield pressure: {options.yield_pressure / 1e3:.1f} kPa")
        print(f"  Density: {options.density:.0f} kg/m³")

    def capture(self):
        """Capture CUDA graph for faster simulation"""
        self.graph = None
        if wp.get_device().is_cuda and self.solver.grid_type == "fixed":
            if self.sim_substeps % 2 != 0:
                wp.utils.warn("Sim substeps must be even for graph capture of MPM step")
            else:
                try:
                    with wp.ScopedCapture() as capture:
                        self.simulate()
                    self.graph = capture.graph
                    print("CUDA graph captured (optimized)")
                except Exception as e:
                    print(f"Graph capture failed: {e}")

    def simulate(self):
        """Run one frame of simulation with substeps"""
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, None, None, self.sim_dt)
            self.solver._project_outside(self.state_1, self.state_1, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        """Advance simulation by one frame"""
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt
        self.current_frame += 1

        # Progress logging
        if self.current_frame % 10 == 0 or self.current_frame == 1:
            positions = self.state_0.particle_q.numpy()
            z_positions = positions[:, 2]
            print(
                f"Frame {self.current_frame:4d}/{self.total_frames}: "
                f"z_avg={z_positions.mean():.3f}m, "
                f"z_min={z_positions.min():.3f}m, "
                f"z_max={z_positions.max():.3f}m"
            )

    def render(self):
        """Render current state to viewer and export USD if enabled"""
        if self.viewer:
            self.viewer.begin_frame(self.sim_time)
            self.viewer.log_state(self.state_0)
            self.viewer.end_frame()

        # Export USD
        if self.export_usd and self.current_frame % self.export_every_n_frames == 0:
            try:
                # Try to use Newton's USD export if available
                usd_path = os.path.join(
                    self.output_dir, f"regolith_frame_{self.current_frame:04d}.usd"
                )
                newton.usd.export_usd(usd_path, self.model, self.state_0)
            except:
                # Fallback: export particle positions as numpy array
                positions = self.state_0.particle_q.numpy()
                npy_path = os.path.join(
                    self.output_dir, f"frame_{self.current_frame:04d}.npy"
                )
                np.save(npy_path, positions)

    def is_complete(self):
        """Check if simulation is complete"""
        return self.current_frame >= self.total_frames


def main():
    """Main entry point"""

    # Create parser with Newton's common arguments
    parser = newton.examples.create_parser()

    # Scene configuration
    parser.add_argument("--domain-x", type=float, default=1.0, help="Domain size X (m)")
    parser.add_argument("--domain-y", type=float, default=1.0, help="Domain size Y (m)")
    parser.add_argument("--domain-z", type=float, default=0.5, help="Domain size Z (m)")
    parser.add_argument(
        "--target-particles", type=int, default=500000, help="Target particle count"
    )
    parser.add_argument(
        "--total-frames", type=int, default=1000, help="Total simulation frames"
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Output frame rate")
    parser.add_argument(
        "--substeps", type=int, default=5, help="Physics substeps per frame"
    )
    parser.add_argument(
        "--gravity",
        type=float,
        nargs=3,
        default=[0, 0, -1.62],
        help="Gravity vector (m/s²)",
    )

    # Material properties (lunar regolith defaults)
    parser.add_argument(
        "--density", type=float, default=1500.0, help="Material density (kg/m³)"
    )
    parser.add_argument(
        "--young-modulus", type=float, default=50e6, help="Young's modulus (Pa)"
    )
    parser.add_argument(
        "--poisson-ratio", type=float, default=0.3, help="Poisson's ratio"
    )
    parser.add_argument(
        "--friction", type=float, default=0.6, help="Friction coefficient"
    )
    parser.add_argument(
        "--yield-pressure", type=float, default=5e3, help="Yield pressure (Pa)"
    )
    parser.add_argument(
        "--yield-stress", type=float, default=1e3, help="Yield stress (Pa)"
    )
    parser.add_argument(
        "--tensile-yield-ratio", type=float, default=0.1, help="Tensile yield ratio"
    )
    parser.add_argument("--hardening", type=float, default=5.0, help="Hardening factor")
    parser.add_argument("--damping", type=float, default=0.05, help="Viscous damping")

    # Solver configuration
    parser.add_argument(
        "--voxel-size", type=float, default=0.02, help="Grid voxel size (m)"
    )
    parser.add_argument(
        "--grid-type", type=str, default="sparse", choices=["sparse", "fixed", "dense"]
    )
    parser.add_argument(
        "--solver", type=str, default="gauss-seidel", choices=["gauss-seidel", "jacobi"]
    )
    parser.add_argument(
        "--transfer-scheme", type=str, default="apic", choices=["apic", "pic"]
    )
    parser.add_argument("--strain-basis", type=str, default="P0", choices=["P0", "Q1"])
    parser.add_argument(
        "--max-iterations", type=int, default=250, help="Solver max iterations"
    )
    parser.add_argument(
        "--tolerance", type=float, default=1e-5, help="Solver tolerance"
    )
    parser.add_argument("--grid-padding", type=int, default=0, help="Grid padding")
    parser.add_argument(
        "--critical-fraction", type=float, default=0.0, help="Critical fraction"
    )
    parser.add_argument(
        "--air-drag", type=float, default=1.0, help="Air drag coefficient"
    )

    # Export options
    parser.add_argument("--export-usd", action="store_true", help="Export USD files")
    parser.add_argument(
        "--output-dir", type=str, default="output", help="Output directory"
    )
    parser.add_argument(
        "--export-every-n-frames", type=int, default=1, help="Export every N frames"
    )

    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init(parser)

    # Create simulation
    sim = LunarRegolithSimulation(viewer, args)

    # Run simulation
    print(f"\nRunning simulation for {args.total_frames} frames...")
    print("-" * 60)

    newton.examples.run(sim, args)

    print("-" * 60)
    print("Simulation complete!")
    if args.export_usd:
        print(f"Output saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
