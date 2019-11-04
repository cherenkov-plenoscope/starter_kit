import numpy as np


class Plenoscope:
    def __init__(self, integrated_lookup, light_field_geometry):
        self.il = integrated_lookup
        self.lfg = light_field_geometry

        self._pixel_radius = (
            1.3*self.lfg.sensor_plane2imaging_system.pixel_FoV_hex_flat2flat/2.)
        _aperture_area = (
            self.lfg.sensor_plane2imaging_system.\
            expected_imaging_system_max_aperture_radius**2*np.pi)
        self._paxel_radius_according_to_lfg = np.sqrt(
            (_aperture_area/self.lfg.number_paxel)/np.pi)

        assert np.abs(
            self._paxel_radius_according_to_lfg -
            self.il.integrated["aperture_bin_radius"]) < (
                0.1*self._paxel_radius_according_to_lfg)

    def lixel_response(
        self,
        source_direction_cx,
        source_direction_cy,
        core_position_x,
        core_position_y,
        energy,
        shower_altitude,
    ):
        c_para = self.il.integrated["c_parallel_bin_centers"]
        c_perp = self.il.integrated["c_perpendicular_bin_centers"]
        pixel_c_para = self.il.integrated["pixel_c_para"]
        pixel_c_perp = self.il.integrated["pixel_c_perp"]

        images_in_paxels = []
        for paxel in range(self.lfg.number_paxel):
            aperture_x = -core_position_x - self.lfg.paxel_pos_x[paxel]
            aperture_y = -core_position_y - self.lfg.paxel_pos_y[paxel]
            paxel_azimuth = np.arctan2(aperture_y, aperture_x)
            paxel_azimuth = np.mod(paxel_azimuth, 2*np.pi)
            assert paxel_azimuth >= 0.
            assert paxel_azimuth < 2*np.pi
            paxel_to_core = np.hypot(aperture_x, aperture_y)

            template_image_in_paxel = self.il.image_interpolate(
                energy=energy,
                altitude=shower_altitude,
                azimuth=paxel_azimuth,
                radius=paxel_to_core
            )*self.lfg.paxel_efficiency_along_pixel[paxel]
            template_intensities = template_image_in_paxel.flatten()

            _cx = (
                np.cos(paxel_azimuth)*pixel_c_para -
                np.sin(paxel_azimuth)*pixel_c_perp)
            _cy = (
                np.sin(paxel_azimuth)*pixel_c_para +
                np.cos(paxel_azimuth)*pixel_c_perp)

            cx = _cx + source_direction_cx
            cy = _cy + source_direction_cy

            distances, pixel_idx = self.lfg.pixel_pos_tree.query(
                np.array([cx ,cy]).T,
                k=1)
            pixel_close_enough = distances <= self._pixel_radius
            pixel_bright_enough = template_intensities > 0.
            pixel_valid = np.logical_and(
                pixel_close_enough,
                pixel_bright_enough)

            paxel_response = {}
            _pixel_indices = pixel_idx[pixel_valid]
            _intensities = template_intensities[pixel_valid]
            for p in range(_pixel_indices.shape[0]):
                _pixel_idx = _pixel_indices[p]
                if _pixel_idx in paxel_response:
                    paxel_response[_pixel_idx] += _intensities[p]
                else:
                    paxel_response[_pixel_idx] = _intensities[p]
            images_in_paxels.append(paxel_response)

        lixel_response = np.zeros(self.lfg.number_pixel*self.lfg.number_paxel)
        for paxel_idx, paxel_response in enumerate(images_in_paxels):
            for pixel_idx in paxel_response:
                lixel_idx = pixel_idx*self.lfg.number_paxel + paxel_idx
                lixel_response[lixel_idx] = paxel_response[pixel_idx]
        return lixel_response


def lixel_intensity_to_photon_lixel_indices(lixel_intensity, num_ph=1e5):
    num_lixel = lixel_intensity.shape[0]
    ph_sum = np.sum(lixel_intensity)
    scale = num_ph/ph_sum
    _lix_intensity = (lixel_intensity * scale).astype(np.int32)
    photon_lixel_indices = []
    for lix_idx in range(num_lixel):
        for _i in range(_lix_intensity[lix_idx]):
            photon_lixel_indices.append(lix_idx)
    return photon_lixel_indices
