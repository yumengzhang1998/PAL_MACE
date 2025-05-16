import yaml
from typing import Union, Any
from copy import deepcopy


class ConfigLoader:
    _cache = None  # static cache shared across instances
    def __init__(self, file_path: str):
        # #print("Loading config...")
        # with open(file_path, 'r') as stream:
        #     self._config = yaml.safe_load(stream)
        # #print("Config loaded.")
        if ConfigLoader._cache is None:
            with open(file_path, "r") as f:
                ConfigLoader._cache = yaml.safe_load(f)
        self._config = ConfigLoader._cache
    def get_config(self):
        return self._config

    def get(self, key: str, default: Any = None):
        if key in self._config:
            return self._config[key]
        else:
            return default

    def __getitem__(self, item: str) -> Any:
        """Access config values using bracket notation."""
        if item == 'metadata':
            # Process and replace the 'metadata' field with updated values
            self._process_meta()

        if item not in self._config:
            raise KeyError(f"Key '{item}' not found in configuration.")

        value = self._config[item]

        # Handle 'inf' keyword
        if value == 'inf':
            return float('inf')

        return value
    
    def _process_meta(self):
        """Processes the 'metadata' field to replace placeholders like 'num_atom' in place."""
        if 'metadata' not in self._config:
            raise KeyError("'metadata' field is not defined in the configuration.")

        num_atom = self._config.get('num_atom', None)
        if num_atom is None:
            raise ValueError("`num_atom` must be defined in the config for dynamic shape substitution.")

        # Update 'metadata' directly within self._config
        for entry in self._config['metadata']:
            if "shape" in entry:
                entry["shape"] = [
                    num_atom if dim == "num_atom" else dim for dim in entry["shape"]
                ]
    def update_from_config(self, params: dict, copy: bool = True) -> dict:
        updated = []
        for key, value in params.items():
            if key in self._config:
                if copy:
                    params[key] = deepcopy(self._config[key])
                else:
                    params[key] = self._config[key]
                updated.append(key)
        print("Updated config: %s" % updated)
        return params

    def fetch_params(self, *items):
        return {item: self._config[item] for item in items if item in self._config}
