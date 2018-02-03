from setuptools import setup, find_packages
requirements = [
    'scipy', 'numpy'
]

setup(
    name="Neural Nets Playground",
    version="1.0.0",
    description="Neural Nets implemented in numpy for teaching purposes.",
    packages=find_packages(),
    install_requires=requirements,
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    # 
)
