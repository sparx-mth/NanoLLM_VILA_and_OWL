from setuptools import setup, find_packages

setup(
    name="nano_owl_tree",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pillow",
        "flask",
        "flask-cors",
        "psycopg2-binary",
    ],
    author="TheAgency@SPARX",
    description="A project for tree-based object detection using NanoOwl on Jetson Orin",
    python_requires=">=3.10",
    entry_points={
        'console_scripts': [
            'nanoowl-service=nano_owl_tree.nanoowl_service:main',
            'nanoowl-tree-demo=nano_owl_tree.tree_demo:main',
        ],
    },
)