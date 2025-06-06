# PyCylinder

A Python package for detecting cylindrical shapes in 3D point clouds using Open3D.

## Features

- Detect cylindrical shapes in 3D point clouds
- Support for both synthetic and real-world data
- Configurable parameters for different use cases
- Visualization of detected cylinders
- Logging system for debugging and analysis

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/amitjoy/CylinderDetection.git
   cd CylinderDetection/pycylinder
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Or install the package in development mode:
   ```bash
   pip install -e .
   ```

## Usage

### Running the Examples

The package includes example scripts to demonstrate its functionality:

1. **Basic Example** (`examples/example.py`):
   ```bash
   python examples/example.py
   ```
   This example demonstrates cylinder detection on synthetic data.

2. **Test Data Example** (`examples/example_testdata.py`):
   ```bash
   python examples/example_testdata.py
   ```
   This example demonstrates cylinder detection on sample mesh data.

### Using in Your Code

```python
from pycylinder import CylinderDetector
from pycylinder.logger import get_logger, LogLevel
import open3d as o3d

# Set up logging
logger = get_logger()
logger.set_console_level(LogLevel.INFO)
logger.set_file_level(LogLevel.DEBUG, 'cylinder_detection.log')

# Create a detector instance
detector = CylinderDetector(
    distance_threshold=0.1,  # Adjust based on your data
    min_radius=0.01,          # Minimum cylinder radius to detect
    max_radius=1.0,           # Maximum cylinder radius to detect
    min_points=100,           # Minimum points to consider a valid cylinder
    min_length=0.1,           # Minimum length of a valid cylinder
    angle_threshold=30,       # Maximum angle between normals in degrees
    max_iterations=1000       # Maximum RANSAC iterations
)

# Load your point cloud (replace with your data)
# point_cloud = o3d.io.read_point_cloud("your_point_cloud.ply")

# Or create a synthetic point cloud for testing
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(np.random.rand(1000, 3))

# Detect cylinders
cylinders = detector.detect(point_cloud)

# Process detected cylinders
for i, cylinder in enumerate(cylinders):
    print(f"Cylinder {i+1}:")
    print(f"  Center: {cylinder.center}")
    print(f"  Axis: {cylinder.axis}")
    print(f"  Radius: {cylinder.radius}")
    print(f"  Length: {cylinder.length}")

# Visualize the results (requires Open3D)
if len(cylinders) > 0:
    o3d.visualization.draw_geometries([point_cloud] + [c.to_mesh() for c in cylinders])
```

## Logging

The package includes a logging system that can be configured to output to both console and file. By default:
- Console output shows INFO level and above
- File output (in `logs/cylinder_debug.log`) shows DEBUG level and above

You can adjust the logging levels:

```python
from pycylinder.logger import get_logger, LogLevel

logger = get_logger()
logger.set_console_level(LogLevel.INFO)  # Show INFO and above in console
logger.set_file_level(LogLevel.DEBUG, 'debug.log')  # Log DEBUG and above to file
```

## Directory Structure

```
pycylinder/
├── __init__.py           # Package initialization
├── detector.py           # Main cylinder detection logic
├── geometry.py           # Geometry primitives (Cylinder, Circle, etc.)
├── logger.py             # Logging configuration
├── utils.py              # Utility functions
├── examples/             # Example scripts
│   ├── example.py
│   └── example_testdata.py
└── requirements.txt      # Dependencies
```

## Dependencies

- Python 3.10+
- numpy
- open3d
- scipy
- matplotlib

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
