import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy scalar/array types."""

    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


