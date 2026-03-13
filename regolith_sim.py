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

            # Position camera for 45° side view facing the spawn point
            # Camera positioned to the side of the domain, looking toward the center
            camera_distance = max(self.domain_x, self.domain_y) * 2.0
            camera_pos = wp.vec3(
                self.domain_x / 2 + camera_distance,  # To the side in +X
                self.domain_y / 2,  # Center Y
                self.domain_z * 2.5,  # Height to see spawn and pile
            )

            # Camera looks toward the center (yaw=180° to look along -X toward center)
            if hasattr(self.viewer, "set_camera"):
                self.viewer.set_camera(camera_pos, pitch=-35.0, yaw=180.0)
                print(
                    f"\nCamera positioned at ({camera_pos[0]:.2f}, {camera_pos[1]:.2f}, {camera_pos[2]:.2f}) "
                    f"looking toward spawn point at center"
                )

        # Try to capture CUDA graph for faster simulation
        self.capture()

        print("\n" + "=" * 60)
        print("Simulation initialized successfully!")
        print("=" * 60)

    def emit_particles(self, builder, options):
        """Create particle grid for regolith falling into container"""

        density = options.density
        particles_per_cell = 3

        # Spawn particles in a funnel above the center of the container
        # Narrow spawn area (25% of domain width) positioned higher for dramatic pour effect
        funnel_width_factor = 0.25  # Spawn region is 25% of domain width
        spawn_height = self.domain_z * 2.5  # 2.5x the domain height above ground

        # Calculate grid dimensions for the NARROW spawn region
        # This maintains proper cell aspect ratios for MPM physics
        funnel_domain_x = self.domain_x * funnel_width_factor
        funnel_domain_y = self.domain_y * funnel_width_factor

        total_cells = self.target_particles / particles_per_cell
        cells_per_side = int(np.cbrt(total_cells))

        # Calculate dimensions for the narrow funnel region
        aspect_xy = funnel_domain_x / funnel_domain_y
        aspect_xz = funnel_domain_x / self.domain_z

        dim_x_funnel = int(cells_per_side * np.cbrt(aspect_xy * aspect_xz))
        dim_y_funnel = int(cells_per_side * np.cbrt(1.0 / aspect_xy))
        dim_z_funnel = int(cells_per_side * np.cbrt(1.0 / aspect_xz))

        # Calculate cell sizes for the funnel (maintaining proper aspect ratios)
        cell_size_x_funnel = funnel_domain_x / dim_x_funnel
        cell_size_y_funnel = funnel_domain_y / dim_y_funnel
        cell_size_z_funnel = self.domain_z / dim_z_funnel
        cell_volume_funnel = (
            cell_size_x_funnel * cell_size_y_funnel * cell_size_z_funnel
        )

        # Recalculate voxel size and mass for funnel
        self.voxel_size = min(
            cell_size_x_funnel, cell_size_y_funnel, cell_size_z_funnel
        )
        radius_funnel = (
            max(cell_size_x_funnel, cell_size_y_funnel, cell_size_z_funnel) * 0.5
        )
        mass_funnel = cell_volume_funnel * density

        print(f"\nPour particle grid:")
        print(f"  Stream dimensions: narrow column for pouring effect")
        print(f"  Target particles: {self.target_particles:,}")
        print(f"  Base spawn height: {spawn_height:.2f}m above ground")

        # Center the spawn region above the container
        spawn_center_x = self.domain_x / 2
        spawn_center_y = self.domain_y / 2

        # POUR SPAWN: Create a narrow stream that pours down like from a funnel
        # Instead of multiple clusters, create one continuous column with extreme randomness

        # Narrow stream dimensions - like a funnel pouring
        stream_width_factor = 0.15  # 15% of domain - narrow stream
        stream_domain_x = self.domain_x * stream_width_factor
        stream_domain_y = self.domain_y * stream_width_factor

        # Calculate cell count for the stream to achieve target particles
        # Make it tall and narrow
        stream_cells_total = self.target_particles / particles_per_cell
        stream_cells_z = int(np.cbrt(stream_cells_total) * 2.5)  # Extra tall
        stream_cells_xy = int(np.sqrt(stream_cells_total / stream_cells_z))

        stream_dim_x = max(3, stream_cells_xy)
        stream_dim_y = max(3, stream_cells_xy)
        stream_dim_z = max(20, stream_cells_z)  # At least 20 layers tall

        # Cell sizes for the narrow stream
        stream_cell_x = stream_domain_x / stream_dim_x
        stream_cell_y = stream_domain_y / stream_dim_y
        stream_cell_z = (self.domain_z * 3) / stream_dim_z  # Stretch vertically

        print(f"\n  Pour spawn: narrow stream")
        print(
            f"  Stream: {stream_domain_x:.2f}m x {stream_domain_y:.2f}m x {self.domain_z * 3:.2f}m"
        )
        print(f"  Grid: {stream_dim_x} x {stream_dim_y} x {stream_dim_z}")

        total_particles_spawned = 0

        # Create ONE continuous stream above the tile center
        # Center it over the small tile
        stream_center_x = self.domain_x / 2
        stream_center_y = self.domain_y / 2
        stream_base_height = spawn_height

        # Use maximum jitter to break up the grid into a fluid-like mass
        builder.add_particle_grid(
            pos=wp.vec3(
                stream_center_x - stream_domain_x / 2,
                stream_center_y - stream_domain_y / 2,
                stream_base_height,
            ),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),  # Start at rest - pure gravity fall
            dim_x=stream_dim_x,
            dim_y=stream_dim_y,
            dim_z=stream_dim_z,
            cell_x=stream_cell_x,
            cell_y=stream_cell_y,
            cell_z=stream_cell_z,
            mass=mass_funnel,
            jitter=8.0 * radius_funnel,  # EXTREME jitter for fluid-like appearance
            radius_mean=radius_funnel,
            flags=newton.ParticleFlags.ACTIVE,
        )

        stream_particles = stream_dim_x * stream_dim_y * stream_dim_z
        total_particles_spawned = stream_particles

        print(f"  Total particles spawned: {total_particles_spawned:,}")
        print(
            f"  Stream height: {stream_base_height:.2f}m to {stream_base_height + stream_dim_z * stream_cell_z:.2f}m"
        )
        print(f"  Pouring onto small tile - will pile up and spill over")

    def add_container(self, builder, options):
        """Add rigid container boundaries"""

        # Container collision properties
        container_cfg = newton.ModelBuilder.ShapeConfig(
            mu=options.friction,
            density=0.0,  # Static
            has_particle_collision=True,
        )

        wall_thickness = 0.02  # 2cm

        # Make floor tile smaller than spawn area so regolith falls off edges
        # Floor is 50% of domain size, centered under spawn
        floor_scale = 0.5
        floor_half_x = self.domain_x * floor_scale / 2
        floor_half_y = self.domain_y * floor_scale / 2

        # Ground plane as explicit collision box - smaller than spawn
        builder.add_shape_box(
            body=-1,
            cfg=container_cfg,
            xform=wp.transform(
                wp.vec3(self.domain_x / 2, self.domain_y / 2, -wall_thickness / 2),
                wp.quat_identity(),
            ),
            hx=floor_half_x,
            hy=floor_half_y,
            hz=wall_thickness / 2,
        )

        print(f"\nContainer:")
        print(f"  Ground: small collision box ({floor_scale * 100:.0f}% of domain)")
        print(f"  Size: {floor_half_x * 2:.2f}m x {floor_half_y * 2:.2f}m")
        print(f"  Regolith will pile on tile and spill over edges")

        if options.walled_container:
            print(f"  Ground + 4 walls (boxed)")
            print(f"  Wall thickness: {wall_thickness * 100:.1f} cm")
        else:
            print(f"  Ground box (open - pile formation)")
            print(f"  Regolith will spread and pile naturally")

    def set_material_properties(self, options):
        """Set per-particle material properties for lunar regolith"""

        num_particles = self.model.particle_count

        # Create index array for all particles
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
        "--walled-container",
        action="store_true",
        help="Use 4-wall container (default is open ground for berming)",
    )
    parser.add_argument(
        "--target-particles", type=int, default=100000, help="Target particle count"
    )
    parser.add_argument(
        "--total-frames", type=int, default=1000, help="Total simulation frames"
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Output frame rate")
    parser.add_argument(
        "--substeps", type=int, default=10, help="Physics substeps per frame"
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
        "--young-modulus", type=float, default=10e6, help="Young's modulus (Pa)"
    )
    parser.add_argument(
        "--poisson-ratio", type=float, default=0.3, help="Poisson's ratio"
    )
    parser.add_argument(
        "--friction", type=float, default=0.85, help="Friction coefficient"
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
    print(f"\nRunning simulation for {args.total_frames} frames at {args.fps} FPS...")
    print("-" * 60)

    newton.examples.run(sim, args)

    print("-" * 60)
    print("Simulation complete!")
    if args.export_usd:
        print(f"Output saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
