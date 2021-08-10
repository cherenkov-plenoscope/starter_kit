Use fermi-py to make some FERMI-LAT performance-figures
-------------------------------------------------------

mkdir fermi
cd fermi

git clone https://github.com/fermiPy/fermipy.git

conda create --name fermipy --channel conda-forge --channel fermi fermitools python=3.7 clhep=2.4.4.1 fermipy
