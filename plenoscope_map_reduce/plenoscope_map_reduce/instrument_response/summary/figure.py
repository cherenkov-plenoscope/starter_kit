import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


CONFIG_16_9 = {
    "rows": 1080,
    "cols": 1920,
    "fontsize": 2,
    "format": "jpg"
}


def figure(config=CONFIG_16_9, dpi=120):
    sc = config['fontsize']
    width = config['cols']/dpi
    height = config['rows']/dpi
    return plt.figure(
        figsize=(width/sc, height/sc),
        dpi=dpi*sc)
