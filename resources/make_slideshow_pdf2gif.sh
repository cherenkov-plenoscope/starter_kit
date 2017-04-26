#!/bin/bash
# Make the cover gif slide show for the Atmospheric Cherenkov Plenoscope (ACP)
# 
# 1) Create a slideshow using e.g. libre office impress and export a PDF 
#    document named show.pdf
#
# 2) Run this script next to the show.pdf

convert -density 600 show.pdf -strip -resize 1920x1080 PNG8:slide-%03d.png
convert -layers OptimizePlus -delay 333 slide-???.png -loop 0 show.gif