# FIRI: Fast Incremental Region Inflation for Path Planning

## Introduction

FIRI (Fast Incremental Region Inflation) is an advanced path planning algorithm for autonomous robots in 3D environments with obstacles. The algorithm works by iteratively computing safe regions through constrained ellipsoids and polytopes, then generating smooth trajectories that avoid obstacles while maintaining smooth direction changes.

## Key Features

- **3D Path Planning**: Generates collision-free paths in three-dimensional space
- **Region-based Planning**: Uses incrementally inflated safe regions to guide path planning
- **Adaptive Smoothing**: Path smoothing with angle constraints to reduce sharp turns
- **Performance Monitoring**: Built-in performance evaluation tools for benchmarking
- **Visualization Tools**: Multiple visualization options (Open3D and Matplotlib)

## Project Structure

```
FIRI/
├── firi/                       # Core FIRI algorithm implementation
│   ├── geometry/               # Geometry utilities (convex polytopes, ellipsoids)
│   ├── planning/               # Path planning algorithms
│   ├── utils/                  # Utility functions
│   └── visualization/          # Visualization tools
├── temp/                       # Temporary data output directory
├── main.py                     # Main execution script
├── analyze_trajectory.py       # Trajectory analysis tools
├── angle_comparison.py         # Path angle comparison script
└── README.md                   # This file
```

## Installation

### Prerequisites

- Python 3.7+
- NumPy
- Matplotlib
- Open3D
- SciPy

### Setup

```bash
# Clone the repository
git clone https://gitee.com/ML-Lab-of-SLU-EE/firi.git
cd firi

# Install dependencies
pip install numpy matplotlib open3d scipy psutil
```

## Usage

### Basic Execution

```bash
python main.py
```

This will:
1. Generate random obstacles in 3D space
2. Plan a collision-free path from start to goal
3. Apply path smoothing to reduce sharp angle changes
4. Visualize the results using both Matplotlib and Open3D
5. Generate performance metrics in the `temp` directory

### Path Analysis

```bash
python analyze_trajectory.py
```

Analyzes a generated path and provides statistics on angles, curvature, and safety.

### Path Angle Comparison

```bash
python angle_comparison.py
```

Compares the original path with the smoothed path, focusing on angle changes.

## Algorithm Details

The FIRI algorithm operates in several steps:

1. **Obstacle Processing**: Converts obstacles into a form suitable for collision checking
2. **Safe Region Computation**: Generates convex polytopes and ellipsoids representing safe regions
3. **Path Planning**: Creates a collision-free path through the safe regions
4. **Path Smoothing**: Applies adaptive smoothing to reduce sharp angle changes
5. **Collision Verification**: Ensures the final path is collision-free

## Performance

The algorithm's performance is measured across various stages:

- Path planning: ~0.08 seconds (core algorithm)
- Path smoothing: ~0.03 seconds
- Total planning time: ~0.15 seconds (excluding visualization)

Performance metrics are automatically saved to the `temp` directory for analysis.

## Visualization

FIRI provides two visualization methods:

1. **Matplotlib**: Static 3D visualization of the path and obstacles
2. **Open3D**: Interactive 3D visualization with inflated obstacles and path segments

## Contributing

Contributions to the FIRI project are welcome! Please feel free to submit issues and pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project was developed by the ML Lab of SLU-EE
