import setuptools

setuptools.setup(
    name='event_look_up_table',
    version="0.0.0",
    author='Sebastian Achim Mueller',
    author_email='sebastian-achim.mueller@mpi-hd.mpg.de',
    description='A look up table for plenoptic events.',
    url='https://github.com/cherenkov-plenoscope/event_look_up_table',
    packages=setuptools.find_packages(),
    install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
    ],
    python_requires='>=3',
)
