#!/usr/bin/env python3
"""
Lunar Regolith MPM Simulation using Newton Physics Engine
Two-way coupling with rigid bodies, optimized for RTX 3090
Stiff granular material properties for realistic regolith dynamics
"""

import newton
import newton.examples
import warp as wp
import numpy as np
import os
import argparse

from newton.solvers import SolverImplicitMPM


@wp.kernel
def compute_body_forces(
    dt: float,
    collider_ids: wp.array(dtype=int),
    collider_impulses: wp.array(dtype=wp.vec3),
    collider_impulse_pos: wp.array(dtype=wp.vec3),
    body_ids: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    body_f: wp.array(dtype=wp.spatial_vector),
):
    """Compute forces applied by sand to rigid bodies.
    Sum the impulses applied on each mpm grid node and convert to
    forces and torques at the body's center of mass.
    """
    i = wp.tid()
    cid = collider_ids[i]
    if cid >= 0 and cid < body_ids.shape[0]:
        body_index = body_ids[cid]
        if body_index == -1:
            return

        f_world = collider_impulses[i] / dt
        X_wb = body_q[body_index]
        X_com = body_com[body_index]
        r = collider_impulse_pos[i] - wp.transform_point(X_wb, X_com)
        tau = wp.cross(r, f_world)
        # spatial_vector takes (angular, linear) = (torque, force)
        wp.atomic_add(body_f, body_index, wp.spatial_vector(tau, f_world))


@wp.kernel
def subtract_body_force(
    dt: float,
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_f: wp.array(dtype=wp.spatial_vector),
    body_inv_inertia: wp.array(dtype=wp.mat33),
    body_inv_mass: wp.array(dtype=float),
    body_q_res: wp.array(dtype=wp.transform),
    body_qd_res: wp.array(dtype=wp.spatial_vector),
):
    """Update the rigid bodies velocity to remove the forces applied by sand at the last step."""
    body_id = wp.tid()
    # Remove previously applied force
    f = body_f[body_id]
    delta_v = dt * body_inv_mass[body_id] * wp.spatial_top(f)
    r = wp.transform_get_rotation(body_q[body_id])
    delta_w = dt * wp.quat_rotate(
        r, body_inv_inertia[body_id] * wp.quat_rotate_inv(r, wp.spatial_bottom(f))
    )
    body_q_res[body_id] = body_q[body_id]
    body_qd_res[body_id] = body_qd[body_id] - wp.spatial_vector(delta_v, delta_w)


class LunarRegolithSimulation:
    """Lunar regolith MPM simulation with two-way rigid body coupling"""

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

        # Check if MuJoCo is available
        self.has_mujoco = False
        try:
            import mujoco

            self.has_mujoco = True
        except ImportError:
            pass

        # ============================================================
        # RIGID BODY MODEL (always has ground plane, optionally has rocks)
        # ============================================================
        rb_builder = newton.ModelBuilder()
        rb_builder.default_shape_cfg.mu = options.ground_friction

        # ALWAYS add ground plane (visible terrain)
        rb_builder.add_ground_plane(
            cfg=newton.ModelBuilder.ShapeConfig(mu=options.ground_friction)
        )

        # Add rigid bodies (rocks) if MuJoCo is available and requested
        if self.has_mujoco and options.add_rigid_bodies:
            self._emit_rigid_bodies(rb_builder, options)
            print("  Added rigid rocks for two-way coupling")
        elif options.add_rigid_bodies and not self.has_mujoco:
            print(
                "  Warning: MuJoCo not installed, skipping rigid bodies. "
                "Install with: uv add mujoco"
            )

        # Finalize rigid body model (always has at least ground plane)
        self.rb_model = rb_builder.finalize()
        self.rb_model.set_gravity(options.gravity)

        # Setup rigid body solver only if we have dynamic bodies (not just ground)
        has_dynamic_bodies = self.has_mujoco and options.add_rigid_bodies
        if has_dynamic_bodies:
            self.rb_solver = newton.solvers.SolverMuJoCo(
                self.rb_model, use_mujoco_contacts=False, njmax=100
            )
            has_rigid_bodies = True
            print("  Rigid body solver: MuJoCo (two-way coupling enabled)")
        else:
            self.rb_solver = None
            has_rigid_bodies = False
            print("  Rigid body solver: None (ground only)")

        # ============================================================
        # SAND/REGOLITH MODEL (MPM particles)
        # ============================================================
        sand_builder = newton.ModelBuilder()

        # Register MPM custom attributes (CRITICAL: before adding particles)
        SolverImplicitMPM.register_custom_attributes(sand_builder)

        # Emit regolith particles
        self._emit_regolith(sand_builder, options)

        # Finalize sand model
        self.sand_model = sand_builder.finalize()
        self.sand_model.set_gravity(options.gravity)

        # Initialize sand state (needed regardless of rigid bodies)
        self.sand_state_0 = self.sand_model.state()

        print(f"\nModel finalized:")
        if has_rigid_bodies:
            print(f"  Rigid bodies: {self.rb_model.body_count}")
        else:
            print(f"  Rigid bodies: 0 (MPM-only mode)")
        print(f"  Sand particles: {self.sand_model.particle_count:,}")
        print(f"  Voxel size: {self.voxel_size:.4f} m")
        print(f"  Gravity: {options.gravity}")

        # Initialize states
        if has_rigid_bodies:
            self.rb_state_0 = self.rb_model.state()
            self.rb_state_1 = self.rb_model.state()
            # Link sand state to rigid body state for two-way coupling
            self.sand_state_0.body_q = wp.empty_like(self.rb_state_0.body_q)
            self.sand_state_0.body_qd = wp.empty_like(self.rb_state_0.body_qd)
            self.sand_state_0.body_f = wp.empty_like(self.rb_state_0.body_f)
            # Controls and contacts
            self.control = self.rb_model.control()
            self.contacts = self.rb_model.contacts()
            device = self.rb_model.device
            self.has_rigid_bodies = True
        else:
            self.rb_state_0 = None
            self.rb_state_1 = None
            self.control = None
            self.contacts = None
            device = self.sand_model.device
            self.has_rigid_bodies = False

        # Configure MPM solver options (stiff granular material)
        mpm_config = SolverImplicitMPM.Config()
        mpm_config.voxel_size = self.voxel_size
        mpm_config.max_iterations = options.max_iterations
        mpm_config.tolerance = options.tolerance  # Tight tolerance for accuracy
        mpm_config.strain_basis = options.strain_basis
        mpm_config.solver = options.solver
        mpm_config.transfer_scheme = options.transfer_scheme  # APIC for less diffusion
        mpm_config.grid_type = options.grid_type
        mpm_config.grid_padding = options.grid_padding
        mpm_config.critical_fraction = options.critical_fraction
        mpm_config.air_drag = options.air_drag

        # Initialize MPM solver
        self.mpm_solver = SolverImplicitMPM(self.sand_model, mpm_config)

        # Setup colliders from rigid body model (if available)
        if has_rigid_bodies:
            self.mpm_solver.setup_collider(model=self.rb_model)
            # Debug: Check if colliders are properly set up
            if hasattr(self.mpm_solver, "collider_body_index"):
                print(
                    f"  Colliders set up: {self.mpm_solver.collider_body_index.shape[0]} bodies"
                )
            else:
                print("  WARNING: No collider_body_index found!")
        else:
            # No rigid bodies, just use ground plane collision
            self.collider_body_id = None

        # Set per-particle material properties (stiff lunar regolith)
        self._set_material_properties(options)

        # Setup viewer
        if viewer:
            if has_rigid_bodies:
                self.viewer.set_model(self.rb_model)  # Show rigid bodies
            else:
                self.viewer.set_model(self.sand_model)  # Show particles only
            self.viewer.show_particles = True

            # Position camera for side view
            camera_distance = max(self.domain_x, self.domain_y) * 2.0
            camera_pos = wp.vec3(
                self.domain_x / 2 + camera_distance,
                self.domain_y / 2,
                self.domain_z * 2.5,
            )

            if hasattr(self.viewer, "set_camera"):
                self.viewer.set_camera(camera_pos, pitch=-35.0, yaw=180.0)

        # Evaluate forward kinematics (required for proper initialization)
        if has_rigid_bodies:
            newton.eval_fk(
                self.rb_model,
                self.rb_model.joint_q,
                self.rb_model.joint_qd,
                self.rb_state_0,
            )

        # Additional buffers for two-way coupling
        max_nodes = 1 << 20
        self.collider_impulses = wp.zeros(max_nodes, dtype=wp.vec3, device=device)
        self.collider_impulse_pos = wp.zeros(max_nodes, dtype=wp.vec3, device=device)
        self.collider_impulse_ids = wp.full(
            max_nodes, value=-1, dtype=int, device=device
        )
        self._collect_collider_impulses()

        # Map from collider index to body index
        if has_rigid_bodies:
            self.collider_body_id = self.mpm_solver.collider_body_index
        else:
            self.collider_body_id = None

        # Per-body forces from sand (only needed for two-way coupling)
        if has_rigid_bodies:
            self.body_sand_forces = wp.zeros_like(self.rb_state_0.body_f)
        else:
            self.body_sand_forces = None

        # Particle render colors
        self.particle_render_colors = wp.full(
            self.sand_model.particle_count,
            value=wp.vec3(0.7, 0.6, 0.4),  # Sandy color
            dtype=wp.vec3,
            device=self.sand_model.device,
        )

        # Try to capture CUDA graph for faster simulation
        # DISABLED: Causing memory issues with two-way coupling
        # self._capture()
        self.graph = None
        print("  CUDA graph capture: Disabled (using normal simulation)")

        print("\n" + "=" * 60)
        print("Simulation initialized successfully!")
        print("Two-way coupling enabled: rigid bodies <-> regolith")
        print("=" * 60)

    def _emit_rigid_bodies(self, builder, options):
        """Add rigid bodies (rocks, tools) that interact with regolith"""

        # Drop some rocks into the sand bed
        drop_z = self.domain_z * 1.5
        offsets_xy = [
            (0.1, 0.0),
            (-0.1, 0.0),
            (0.0, 0.1),
            (0.0, -0.1),
            (0.15, 0.15),
            (-0.15, -0.15),
        ]

        z_separation = 0.3

        # Add various sized rocks
        rocks = [
            (0.08, 0.06, 0.05),  # Small rock
            (0.10, 0.08, 0.06),  # Medium rock
            (0.06, 0.05, 0.04),  # Small rock
            (0.12, 0.10, 0.08),  # Large rock
        ]

        for i, rock in enumerate(rocks):
            (hx, hy, hz) = rock
            ox, oy = offsets_xy[i % len(offsets_xy)]
            pz = drop_z + float(i) * z_separation

            body = builder.add_body(
                xform=wp.transform(
                    p=wp.vec3(
                        self.domain_x / 2 + float(ox), self.domain_y / 2 + float(oy), pz
                    ),
                    q=wp.normalize(wp.quatf(0.0, 0.0, 0.0, 1.0)),
                ),
                mass=5.0,  # 5 kg rocks
            )
            # Add collision shape with explicit particle collision enabled
            builder.add_shape_box(
                body,
                hx=float(hx),
                hy=float(hy),
                hz=float(hz),
                cfg=newton.ModelBuilder.ShapeConfig(
                    mu=options.friction,
                    has_particle_collision=True,  # CRITICAL: Enable particle collision
                ),
            )

        print(f"  Added {len(rocks)} rigid rocks")

    def _emit_regolith(self, builder, options):
        """Create regolith particle bed with proper granular structure"""

        density = options.density
        particles_per_cell = 3

        # Spawn particles in a block above the ground
        bed_lo = np.array(
            [
                self.domain_x / 2 - options.bed_width / 2,
                self.domain_y / 2 - options.bed_depth / 2,
                0.0,
            ]
        )
        bed_hi = np.array(
            [
                self.domain_x / 2 + options.bed_width / 2,
                self.domain_y / 2 + options.bed_depth / 2,
                options.bed_height,
            ]
        )

        # Calculate grid resolution
        bed_res = np.array(
            np.ceil(particles_per_cell * (bed_hi - bed_lo) / self.voxel_size),
            dtype=int,
        )

        cell_size = (bed_hi - bed_lo) / bed_res
        cell_volume = np.prod(cell_size)
        radius = float(np.max(cell_size) * 0.5)
        mass = float(cell_volume * density)

        print(f"\nRegolith particle bed:")
        print(
            f"  Dimensions: {options.bed_width:.2f}m x {options.bed_depth:.2f}m x {options.bed_height:.2f}m"
        )
        print(f"  Grid resolution: {bed_res[0]} x {bed_res[1]} x {bed_res[2]}")
        print(
            f"  Cell size: {cell_size[0]:.3f} x {cell_size[1]:.3f} x {cell_size[2]:.3f} m"
        )
        print(f"  Target particles: {self.target_particles:,}")

        builder.add_particle_grid(
            pos=wp.vec3(bed_lo),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=bed_res[0] + 1,
            dim_y=bed_res[1] + 1,
            dim_z=bed_res[2] + 1,
            cell_x=cell_size[0],
            cell_y=cell_size[1],
            cell_z=cell_size[2],
            mass=mass,
            jitter=2.0 * radius,  # Moderate jitter (like granular example)
            radius_mean=radius,
            custom_attributes={"mpm:friction": options.friction},
        )

    def _set_material_properties(self, options):
        """Set per-particle material properties for stiff granular regolith"""

        num_particles = self.sand_model.particle_count

        # Create index array for all particles
        indices = wp.array(
            np.arange(num_particles), dtype=int, device=self.sand_model.device
        )

        # Set stiff granular material properties
        self.sand_model.mpm.young_modulus[indices].fill_(options.young_modulus)
        self.sand_model.mpm.poisson_ratio[indices].fill_(options.poisson_ratio)
        self.sand_model.mpm.friction[indices].fill_(options.friction)
        self.sand_model.mpm.yield_pressure[indices].fill_(options.yield_pressure)
        self.sand_model.mpm.yield_stress[indices].fill_(options.yield_stress)
        self.sand_model.mpm.tensile_yield_ratio[indices].fill_(
            options.tensile_yield_ratio
        )
        self.sand_model.mpm.hardening[indices].fill_(options.hardening)
        self.sand_model.mpm.damping[indices].fill_(options.damping)

        print(f"\nMaterial properties (stiff granular regolith):")
        print(f"  Young's modulus: {options.young_modulus:.1e} Pa")
        print(f"  Poisson ratio: {options.poisson_ratio:.2f}")
        print(
            f"  Friction: {options.friction:.2f} (~{np.degrees(np.arctan(options.friction)):.1f}°)"
        )
        print(f"  Yield pressure: {options.yield_pressure:.1e} Pa")
        print(f"  Density: {options.density:.0f} kg/m³")
        print(f"  Transfer scheme: {options.transfer_scheme.upper()}")

    def _collect_collider_impulses(self):
        """Collect impulses from sand colliders for two-way coupling"""
        impulses, pos, ids = self.mpm_solver._collect_collider_impulses(
            self.sand_state_0
        )
        self.collider_impulse_ids.fill_(-1)
        n_colliders = min(impulses.shape[0], self.collider_impulses.shape[0])
        self.collider_impulses[:n_colliders].assign(impulses[:n_colliders])
        self.collider_impulse_pos[:n_colliders].assign(pos[:n_colliders])
        self.collider_impulse_ids[:n_colliders].assign(ids[:n_colliders])

    def _capture(self):
        """Capture CUDA graph for faster simulation"""
        self.graph = None
        if wp.get_device().is_cuda and self.mpm_solver.grid_type == "fixed":
            if self.sim_substeps % 2 != 0:
                wp.utils.warn("Sim substeps must be even for graph capture of MPM step")
            else:
                try:
                    with wp.ScopedCapture() as capture:
                        self._simulate()
                    self.graph = capture.graph
                    print("CUDA graph captured (optimized)")
                except Exception as e:
                    print(f"Graph capture failed: {e}")

    def _simulate(self):
        """Run one frame of simulation with optional two-way coupling"""
        # Run all rigid body substeps first (like Newton example)
        for _ in range(self.sim_substeps):
            if self.has_rigid_bodies:
                # Clear forces on rigid bodies
                self.rb_state_0.clear_forces()

                # Apply forces from sand to rigid bodies
                wp.launch(
                    compute_body_forces,
                    dim=self.collider_impulse_ids.shape[0],
                    inputs=[
                        self.frame_dt,
                        self.collider_impulse_ids,
                        self.collider_impulses,
                        self.collider_impulse_pos,
                        self.collider_body_id,
                        self.rb_state_0.body_q,
                        self.rb_model.body_com,
                        self.rb_state_0.body_f,
                    ],
                )

                # Save applied forces
                self.body_sand_forces.assign(self.rb_state_0.body_f)

                # Apply external forces via viewer
                if self.viewer:
                    self.viewer.apply_forces(self.rb_state_0)

                # Rigid body collision detection
                self.rb_model.collide(self.rb_state_0, self.contacts)

                # Step rigid body solver (MuJoCo)
                self.rb_solver.step(
                    self.rb_state_0,
                    self.rb_state_1,
                    self.control,
                    self.contacts,
                    self.sim_dt,
                )

                # Swap rigid body states
                self.rb_state_0, self.rb_state_1 = self.rb_state_1, self.rb_state_0

        # Simulate sand ONCE after all rigid body substeps (like Newton example)
        self._simulate_sand()

    def _simulate_sand(self):
        """Simulate sand particles with optional two-way coupling"""
        # Subtract previously applied impulses from body velocities
        if self.has_rigid_bodies and self.sand_state_0.body_q is not None:
            wp.launch(
                subtract_body_force,
                dim=self.sand_state_0.body_q.shape,
                inputs=[
                    self.frame_dt,
                    self.rb_state_0.body_q,
                    self.rb_state_0.body_qd,
                    self.body_sand_forces,
                    self.rb_model.body_inv_inertia,
                    self.rb_model.body_inv_mass,
                    self.sand_state_0.body_q,
                    self.sand_state_0.body_qd,
                ],
            )

        # Step MPM solver
        self.mpm_solver.step(
            self.sand_state_0,
            self.sand_state_0,
            contacts=None,
            control=None,
            dt=self.frame_dt,
        )

        # Save impulses for next rigid body step (collect AFTER MPM step)
        if self.has_rigid_bodies:
            self._collect_collider_impulses()

    def step(self):
        """Advance simulation by one frame"""
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self._simulate()

        self.sim_time += self.frame_dt
        self.current_frame += 1

        # Progress logging
        if self.current_frame % 10 == 0 or self.current_frame == 1:
            positions = self.sand_state_0.particle_q.numpy()
            z_positions = positions[:, 2]

            # Also log rock positions if using rigid bodies
            if self.has_rigid_bodies:
                # Get rock z positions from transform
                rock_transforms = self.rb_state_0.body_q.numpy()
                rock_z = []
                for i in range(min(4, rock_transforms.shape[0])):
                    # Extract z from 4x4 transform matrix (index 14 is z translation)
                    z = (
                        rock_transforms[i][2, 3]
                        if len(rock_transforms[i].shape) > 1
                        else rock_transforms[i][2]
                    )
                    rock_z.append(z)

                print(
                    f"Frame {self.current_frame:4d}/{self.total_frames}: "
                    f"sand_z_avg={z_positions.mean():.3f}m, "
                    f"rocks_z=[{', '.join([f'{z:.2f}' for z in rock_z])}]m"
                )
            else:
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

            # Render rigid bodies (if using two-way coupling)
            if self.has_rigid_bodies:
                self.viewer.log_state(self.rb_state_0)
                self.viewer.log_contacts(self.contacts, self.rb_state_0)

            # Render sand particles
            self.viewer.log_points(
                "/sand",
                points=self.sand_state_0.particle_q,
                radii=self.sand_model.particle_radius,
                colors=self.particle_render_colors,
                hidden=not self.viewer.show_particles,
            )

            self.viewer.end_frame()

        # Export USD
        if self.export_usd and self.current_frame % self.export_every_n_frames == 0:
            try:
                usd_path = os.path.join(
                    self.output_dir, f"regolith_frame_{self.current_frame:04d}.usd"
                )
                if self.has_rigid_bodies:
                    newton.usd.export_usd(usd_path, self.rb_model, self.rb_state_0)
                else:
                    # Fallback: export sand model instead
                    newton.usd.export_usd(usd_path, self.sand_model, self.sand_state_0)
            except:
                # Fallback: export particle positions as numpy array
                positions = self.sand_state_0.particle_q.numpy()
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
    parser.add_argument("--domain-x", type=float, default=2.0, help="Domain size X (m)")
    parser.add_argument("--domain-y", type=float, default=2.0, help="Domain size Y (m)")
    parser.add_argument("--domain-z", type=float, default=1.0, help="Domain size Z (m)")
    parser.add_argument(
        "--target-particles", type=int, default=100000, help="Target particle count"
    )
    parser.add_argument(
        "--total-frames", type=int, default=1000, help="Total simulation frames"
    )
    parser.add_argument("--fps", type=float, default=60.0, help="Output frame rate")
    parser.add_argument(
        "--substeps",
        type=int,
        default=8,
        help="Physics substeps per frame (higher=more stable)",
    )
    parser.add_argument(
        "--gravity",
        type=float,
        nargs=3,
        default=[0, 0, -1.62],
        help="Gravity vector (m/s²)",
    )

    # Bed dimensions
    parser.add_argument(
        "--bed-width", type=float, default=1.0, help="Sand bed width (m)"
    )
    parser.add_argument(
        "--bed-depth", type=float, default=1.0, help="Sand bed depth (m)"
    )
    parser.add_argument(
        "--bed-height", type=float, default=0.5, help="Sand bed height (m)"
    )

    # Rigid body options
    parser.add_argument(
        "--add-rigid-bodies", action="store_true", help="Add rigid rocks/bodies"
    )
    parser.add_argument(
        "--ground-friction", type=float, default=0.5, help="Ground friction"
    )

    # Material properties (LUNAR REGOLITH - stable for simulation)
    # Realistic values based on lunar soil properties
    parser.add_argument(
        "--density", type=float, default=1500.0, help="Material density (kg/m³)"
    )
    parser.add_argument(
        "--young-modulus",
        type=float,
        default=1.0e7,  # 10 MPa - realistic for compacted regolith (was 1e15)
        help="Young's modulus (Pa)",
    )
    parser.add_argument(
        "--poisson-ratio",
        type=float,
        default=0.3,  # Like the granular example
        help="Poisson's ratio",
    )
    parser.add_argument(
        "--friction",
        type=float,
        default=0.6,  # Realistic for lunar regolith (~31°)
        help="Friction coefficient",
    )
    parser.add_argument(
        "--yield-pressure",
        type=float,
        default=1.0e4,  # 10 kPa - realistic yield stress (was 1e12)
        help="Yield pressure (Pa)",
    )
    parser.add_argument(
        "--yield-stress", type=float, default=0.0, help="Yield stress (Pa)"
    )
    parser.add_argument(
        "--tensile-yield-ratio",
        type=float,
        default=0.0,
        help="Tensile yield ratio (0 for dry sand)",
    )
    parser.add_argument(
        "--hardening",
        type=float,
        default=0.0,
        help="Hardening factor (0 for loose sand)",
    )
    parser.add_argument(
        "--damping",
        type=float,
        default=0.0,
        help="Viscous damping",
    )

    # Solver configuration (matching granular example)
    parser.add_argument(
        "--voxel-size", type=float, default=0.05, help="Grid voxel size (m)"
    )
    parser.add_argument(
        "--grid-type", type=str, default="fixed", choices=["sparse", "fixed", "dense"]
    )
    parser.add_argument(
        "--solver", type=str, default="gauss-seidel", choices=["gauss-seidel", "jacobi"]
    )
    parser.add_argument(
        "--transfer-scheme",
        type=str,
        default="apic",  # APIC for less diffusion (like granular example)
        choices=["apic", "pic"],
        help="Particle transfer scheme",
    )
    parser.add_argument("--strain-basis", type=str, default="P0", choices=["P0", "Q1"])
    parser.add_argument(
        "--max-iterations", type=int, default=50, help="Solver max iterations"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1.0e-6,  # Tight tolerance like granular example
        help="Solver tolerance",
    )
    parser.add_argument("--grid-padding", type=int, default=50, help="Grid padding")
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
