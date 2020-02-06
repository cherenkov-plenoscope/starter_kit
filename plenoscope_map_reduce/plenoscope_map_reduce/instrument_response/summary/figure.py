import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


CONFIG = {
    "rows": 1080,
    "cols": 1920,
    "fontsize": 2,
    "format": "jpg"
}


def figure(config=CONFIG, dpi=120):
    sc = config['fontsize']
    width = config['cols']/dpi
    height = config['rows']/dpi
    return plt.figure(
        figsize=(width/sc, height/sc),
        dpi=dpi*sc)
