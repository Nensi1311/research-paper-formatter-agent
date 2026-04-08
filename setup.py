"""Setup file for backwards compatibility with older build tools."""
from setuptools import setup, find_packages

setup(
    name="scholar-env",
    version="0.4.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "scholar-env-server=server.app:main",
            "serve=server.app:main",
            "start=server.app:main",
        ],
    },
)
