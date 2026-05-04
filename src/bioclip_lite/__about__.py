# Single source of truth for package identity.
# pyproject.toml reads __version__ via hatchling's dynamic version
# (see [tool.hatch.version]); image_service.py reads all three for the User-Agent.
__title__ = "bioclip-image-search-lite"
__version__ = "0.1.0"
__url__ = "https://github.com/Imageomics/bioclip-image-search-lite"
