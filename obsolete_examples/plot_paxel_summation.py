import plenopy as pl
import matplotlib.pyplot as plt

light_field_geometry = pl.LightFieldGeometry('./run/light_field_calibration/')

paxel_id = 54

def make_paxel_mask(paxel_id, light_field_geometry):
    mask = np.zeros(light_field_geometry.number_lixel, dtype=np.bool)
    for lixel in range(light_field_geometry.number_lixel):
        pixel = lixel//light_field_geometry.number_paxel
        paxel = lixel - pixel*light_field_geometry.number_paxel
        if paxel == paxel_id:
            mask[lixel] = True
    return mask


def make_pixel_mask(pixel_id, light_field_geometry):
    mask = np.zeros(light_field_geometry.number_lixel, dtype=np.bool)
    for lixel in range(light_field_geometry.number_lixel):
        pixel = lixel//light_field_geometry.number_paxel
        if pixel == pixel_id:
            mask[lixel] = True
    return mask


mask = make_paxel_mask(
    paxel_id=paxel_id,
    light_field_geometry=light_field_geometry)


fsize = pl.plot.FigureSize(
    relative_width=1,
    relative_hight=1,
    pixel_rows=1080,
    dpi=200)

fig = plt.figure(figsize=(fsize.width, fsize.hight), dpi=fsize.dpi)
ax = plt.gca()
pl.plot.light_field_geometry.colored_lixels(
    lss=light_field_geometry,
    I=mask,
    ax=ax,
    cmap='binary',
    edgecolors='k')
ax.set_xlabel('photo-sensor-plane-x/m')
ax.set_ylabel('photo-sensor-plane-y/m')
ax.set_xlim([-.35,.35])
ax.set_ylim([-.35,.35])
plt.savefig(
    'paxel_id_{:d}_summation.png'.format(paxel_id),
    dpi=fsize.dpi,
    transparent=False)
plt.close('all')


pixel_id = light_field_geometry.number_pixel//2
mask = make_pixel_mask(
    pixel_id=pixel_id,
    light_field_geometry=light_field_geometry)


fig = plt.figure(figsize=(fsize.width, fsize.hight), dpi=fsize.dpi)
ax = plt.gca()
pl.plot.light_field_geometry.colored_lixels(
    lss=light_field_geometry,
    I=mask,
    ax=ax,
    cmap='binary',
    edgecolors='k')
ax.set_xlabel('photo-sensor-plane-x/m')
ax.set_ylabel('photo-sensor-plane-y/m')
ax.set_xlim([-.35,.35])
ax.set_ylim([-.35,.35])
plt.savefig(
    'pixel_id_{:d}_summation.png'.format(pixel_id),
    dpi=fsize.dpi,
    transparent=False)
plt.close('all')