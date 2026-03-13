# Lunar Regolith MPM Simulation

Material Point Method simulation of lunar regolith using [Newton](https://github.com/NVIDIA-Omniverse/Newton) and NVIDIA Warp on RTX 3090.

## Quick Start

```bash
# Install dependencies
uv sync

# Run simulation
uv run python regolith_sim.py
```

## What's This?

Simulates granular material (lunar regolith) falling onto a surface using:
- **100K particles** with lunar material properties
- **Implicit MPM solver** (sparse grid, APIC transfer)
- **Lunar gravity** (1.62 m/s²)
- **RTX 3090 optimized** (~5-10 FPS with viewer)

Particles spawn above the surface and fall naturally, forming piles/berms based on friction and material properties. See [SAMPLE.md](SAMPLE.md) for a minimal code example.

## Configuration

All parameters are CLI arguments:

```bash
# Quick test (10K particles, 100 frames)
uv run python regolith_sim.py --target-particles 10000 --total-frames 100

# With USD export (berm formation on open ground)
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
| `--walled-container` | False | Use 4-wall container (default: open ground for berming) |
| `--voxel-size` | 0.02 | Grid resolution (m) |
| `--young-modulus` | 50e6 | Material stiffness (Pa) |
| `--friction` | 0.6 | Friction coefficient (~35°) |

Full list: `uv run python regolith_sim.py --help`

## Output

- **NPY files**: Particle positions per frame (`output/frame_XXXX.npy`)
- **USD files**: Optional USD export for Houdini/Blender

## Requirements

- NVIDIA GPU with CUDA (tested on RTX 3090)
- Python 3.10+
- See `pyproject.toml` for dependencies

## License

Apache 2.0 - See [LICENSE](LICENSE)
