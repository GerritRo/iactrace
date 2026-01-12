import yaml # type: ignore[import-untyped]
import pathlib
import numpy as np

class TelescopeConfigBuilder:
    def __init__(self, name, units='m'):
        self.config = {
            'telescope': {
                'name': name,
                'units': units
            },
            'mirror_templates': {},
            'mirrors': [],
            'obstructions': [],
            'sensors': []
        }
    
    def add_mirror_template(self, name, curvature, conic, aspheric_coeffs):
        """Add a mirror surface template"""
        self.config['mirror_templates'][name] = {
            'surface': {
                'curvature': float(curvature),
                'conic': float(conic),
                'aspheric': [float(a) for a in aspheric_coeffs]
            }
        }
        return self
    
    def add_mirror_circular(self, mirror_id, template, position, orientation, radius):
        """Add circular mirror"""
        self.config['mirrors'].append({
            'id': mirror_id,
            'template': template,
            'position': [float(x) for x in position],
            'orientation': [float(x) for x in orientation],
            'aperture': {
                'type': 'circular',
                'radius': float(radius)
            }
        })
        return self
    
    def add_mirror_polygon(self, mirror_id, template, position, orientation, vertices):
        """Add polygon mirror with vertices in local coordinates"""
        self.config['mirrors'].append({
            'id': mirror_id,
            'template': template,
            'position': [float(x) for x in position],
            'orientation': [float(x) for x in orientation],
            'aperture': {
                'type': 'polygon',
                'vertices': [[float(x), float(y)] for x, y in vertices]
            }
        })
        return self
    
    def add_obstruction_box(self, obs_id, p1, p2):
        """Add box obstruction"""
        self.config['obstructions'].append({
            'id': obs_id,
            'type': 'box',
            'p1': [float(x) for x in p1],
            'p2': [float(x) for x in p2]
        })
        return self
    
    def add_obstruction_cylinder(self, obs_id, p1, p2, radius):
        """Add cylinder obstruction"""
        self.config['obstructions'].append({
            'id': obs_id,
            'type': 'cylinder',
            'p1': [float(x) for x in p1],
            'p2': [float(x) for x in p2],
            'r': float(radius)
        })
        return self
    
    def add_square_sensor_array(self, sensor_id, position, orientation, 
                                width, height, bounds):
        """Add sensor with square pixels"""
        self.config['sensors'].append({
            'id': sensor_id,
            'type': 'square',
            'position': [float(x) for x in position],
            'orientation': [float(x) for x in orientation],
            'width': int(width),
            'height': int(height),
            'bounds': [float(x) for x in bounds]
        })
        return self
    
    def add_hexagon_sensor_array(self, sensor_id, position, orientation,
                                 pixel_x, pixel_y):
        """Add sensor with hexagonal pixels"""
        self.config['sensors'].append({
            'id': sensor_id,
            'type': 'hexagonal',
            'position': [float(x) for x in position],
            'orientation': [float(x) for x in orientation],
            'centers_x': [float(x) for x in pixel_x],
            'centers_y': [float(x) for x in pixel_y],
        })
        return self
    
    def save(self, filename, precision=4):
        """Write to YAML file with controlled precision"""

        # Delete if already exists
        filepath = pathlib.Path(filename)
        if filepath.exists():
            filepath.unlink()
            print(f"Deleted existing {filename}")

        # Custom float representer
        def float_rep(dumper, value):
            return dumper.represent_scalar('tag:yaml.org,2002:float', 
                                          f'{value:.{precision}f}')

        yaml.add_representer(float, float_rep)
        yaml.add_representer(np.float64, float_rep)
        yaml.add_representer(np.float32, float_rep)

        with open(filename, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)

        print(f"Saved telescope config to {filename}")