from glob import glob
from setuptools import setup


package_name = "yolo_slam_landing"

setup(
    name=package_name,
    version="0.1.0",
    packages=[
        package_name,
        "yolo_detector",
        "slam_module",
        "fusion",
        "landing_controller",
        "evaluation",
        "pybullet_sim",
    ],
    package_dir={
        "": "src",
    },
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.py")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="yolo_slam_landing",
    maintainer_email="noreply@example.com",
    description="YOLO + SLAM safe-zone landing scaffold.",
    license="Apache-2.0",
)
