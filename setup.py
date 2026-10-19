from setuptools import find_packages, setup

setup(
    name="aniqr_ai",
    version="0.1.0",
    author="AniQR Team",
    description="Anime QR Code Generator using Diffusion and ControlNet",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    # The source uses PEP 604 union types, which require Python 3.10+.
    python_requires=">=3.10",
    install_requires=[
        "qrcode[pil]>=7.4",
        "Pillow>=9.0",
        "opencv-python-headless>=4.5",
        "numpy>=1.20",
        "pyyaml>=6.0",
    ],
)
