import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name='plenoscope_map_reduce',
    version='0.0.0',
    description='Collection of map and reduce tools for parralel computing.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/cherenkov-plenoscope',
    author='Sebastian Achim Mueller',
    author_email='sebastian-achim.mueller@mpi-hd.mpg.de',
    license='GPL v3',
    packages=[
        'plenoscope_map_reduce',
    ],
)
