import newton as nt
import warp as wp

wp.init()
sim = nt.Simulator(dt=1e-3, substeps=10)  # Stable timestep for elastoplastic dynamics

mpm = nt.MPMImplicitSolver()  # Implicit for stability under stiff material
sim.add_solver(mpm)

# Elastoplastic material calibrated to lunar regolith
material = nt.ElastoplasticMaterial(
    E=50e6,  # Young's modulus (Pa) - mid-range for compacted regolith
    nu=0.3,  # Poisson's ratio
    friction_angle=35.0,  # Degrees - lunar angle of repose ~30-45°
    cohesion=0.0,  # Loose, uncohesive dust
    density=1500.0,  # kg/m³ - loose upper regolith
)
mpm.add_particles(material)

# 1M particles in a 1m x 1m x 0.5m bed
num_particles = 1_000_000
positions = wp.array(...)  # Uniform or random sampling within volume
velocities = wp.zeros_like(positions)
particle_volume = 1e-6
masses = wp.full_like(positions[:, 0], material.density * particle_volume)
mpm.add_particles(positions, velocities, masses)

sim.gravity = wp.vec3(0.0, -1.62, 0.0)  # Lunar gravity

# Rigid boundary - extend this to add rover wheels, lander pads, excavator blades
container = nt.RigidBody(shape=nt.Box(size=(1.0, 1.0, 0.5)))
sim.add_rigid_body(container)

# Run and export
for frame in range(1000):
    sim.step()
    nt.export_usd(sim.state, f"regolith_frame_{frame}.usd")
