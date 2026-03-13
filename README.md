# Lunar Regolith MPM Simulation

Material Point Method simulation of lunar regolith pouring onto a surface using [Newton](https://github.com/NVIDIA-Omniverse/Newton) and NVIDIA Warp on RTX 3090.

## Quick Start

```bash
# Install dependencies
uv sync

# Run simulation
uv run python regolith_sim.py
```

## What's This?

Simulates granular material (lunar regolith) pouring from a narrow stream onto a small tile using:
- **~100K particles** spawned in a continuous narrow column
- **Implicit MPM solver** (sparse grid, APIC transfer)
- **Lunar gravity** (1.62 m/s²)
- **Softer material** (10 MPa Young's modulus for better flow)
- **High friction** (0.85 coefficient for chaotic collisions)
- **45° side view camera** to observe piling and spillover
- **RTX 3090 optimized** (~5-10 FPS with viewer)

Particles spawn as a narrow stream above a small tile (50% of domain size), pour down naturally under lunar gravity, pile up on the tile, and spill over the edges. The camera is positioned at a 45° side angle to capture the dynamic flow and piling behavior.

See [SAMPLE.md](SAMPLE.md) for a minimal code example.

## Configuration

All parameters are CLI arguments:

```bash
# Quick test (10K particles, 100 frames)
uv run python regolith_sim.py --target-particles 10000 --total-frames 100

# With USD export (pile formation on small tile)
uv run python regolith_sim.py --export-usd --output-dir ./frames

# Walled container (boxed)
uv run python regolith_sim.py --walled-container

# Headless batch mode
uv run python regolith_sim.py --headless --total-frames 1000
```

### Key Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--target-particles` | 100000 | Particle count |
| `--total-frames` | 1000 | Simulation length |
| `--fps` | 30.0 | Output frame rate |
| `--substeps` | 10 | Physics substeps per frame |
| `--walled-container` | False | Use 4-wall container (default: small tile with spillover) |
| `--voxel-size` | 0.02 | Grid resolution (m) |
| `--young-modulus` | 10e6 | Material stiffness - lower for better flow (Pa) |
| `--friction` | 0.85 | Friction coefficient - higher for chaotic collisions (~40°) |

Full list: `uv run python regolith_sim.py --help`

## Output

- **Console**: Real-time frame progress with particle height statistics
- **NPY files**: Particle positions per frame (`output/frame_XXXX.npy`)
- **USD files**: Optional USD export for Houdini/Blender

## Requirements

- NVIDIA GPU with CUDA (tested on RTX 3090)
- Python 3.10+
- See `pyproject.toml` for dependencies

## Implementation Details

### Spawn Method
Particles spawn as a single continuous narrow stream (15% of domain width) with extreme jitter (8x radius) to create a fluid-like pour effect rather than organized sheets.

### Camera View
45° side view positioned to capture the full stream impact and piling behavior on the small tile.

### Physics Tuning
- **10 substeps**: Better collision resolution
- **10 MPa Young's modulus**: Softer material for improved flow
- **0.85 friction**: High friction angle (~40°) creates chaotic particle collisions
- **PIC transfer scheme**: Better diffusion for granular materials

## License

Apache 2.0 - See [LICENSE](LICENSE)
