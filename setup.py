from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="perception-nassir",  # Replace with your own package name
    version="0.0.4",  # Initial version
    author="Nassir Mohammad",
    # author_email="your.email@example.com",
    description="A method for detecting anomalies in univariate and multivariate data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/M-Nassir/perception",  # GitHub repo URL
    packages=['perception_nassir'],  # Automatically finds packages
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",  # Choose the appropriate license
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',  # numpy package
        'scipy',  # scipy package
    ],
)
