import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="latexifier",
    version="1.0.8",
    author="Guillaume Garrigos",
    author_email="guillaume.garrigos@lpsm.paris",
    license="MIT",
    description="A package to convert python objects into latex strings",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Guillaume-Garrigos/latexify",
    packages=['latexifier', ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        'numpy',
        'sympy',
        'mpmath',
    ],
    include_package_data=True,
)