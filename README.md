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

Simulates granular material (lunar regolith) in a 1m × 1m × 0.5m container using:
- **50K particles** with lunar material properties
- **Implicit MPM solver** (sparse grid, APIC transfer)
- **Lunar gravity** (1.62 m/s²)
- **RTX 3090 optimized** (~5-10 FPS with viewer)

See [SAMPLE.md](SAMPLE.md) for a minimal code example.

## Configuration

All parameters are CLI arguments:

```bash
# Quick test (10K particles, 100 frames)
uv run python regolith_sim.py --target-particles 10000 --num-frames 100

# With USD export
uv run python regolith_sim.py --export-usd --output-dir ./frames

# Headless batch mode
uv run python regolith_sim.py --headless --num-frames 1000
```

### Key Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--target-particles` | 500000 | Particle count |
| `--num-frames` | 100 | Simulation length |
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
