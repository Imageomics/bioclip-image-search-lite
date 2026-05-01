# Single source of truth for the package version.
# pyproject.toml reads this via hatchling's dynamic version
# (see [tool.hatch.version]); image_service.py reads it for the User-Agent.
__version__ = "0.1.0"
