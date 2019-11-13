import numpy as np
import event_look_up_table as elut
import plenopy as pl
import iminuit
import time

d2r = np.deg2rad

event_list_path = "run20190724_10/irf/gamma/past_trigger/"

G = elut.integrated.Reader("lookup5_integrated")
lfg = pl.LightFieldGeometry("run20190724_10/light_field_calibration/")

P = elut.plenoscope.Plenoscope(G, lfg)
event_list = pl.Run(event_list_path, light_field_geometry=lfg)


integration_time = 5e-9
num_time_slices_integral = 10
avg_nsb_rate_in_lixel = 28.05e6
nsb_in_lixel = avg_nsb_rate_in_lixel*integration_time


def estimate_chi2(light_field_measured, light_field_template, nsb_in_lixel):
    lfm = light_field_measured
    lft = light_field_template
    sigma_squares = 1.0*lfm + nsb_in_lixel**2
    summands = ((lfm - lft)**2)/sigma_squares
    chi_square = np.sum(summands)
    return chi_square


for event in event_list:

    print(event)
    if event.cherenkov_photons.number < 100:
        print("too dim")
        continue

    radial_angle_center_of_light = np.hypot(
        np.median(event.cherenkov_photons.cx),
        np.median(event.cherenkov_photons.cy))

    if radial_angle_center_of_light > d2r(2.25):
        print("too far out")
        continue

    # find time-window
    trigger_response = pl.trigger.read_trigger_response_of_event(event)
    time_slice_max_active = []
    for refocus_response in trigger_response:
        time_slice_max_active.append(refocus_response[
            "time_slice_with_most_active_neighboring_patches"])
    time_slice_max_active = int(np.median(time_slice_max_active))

    time_slice_start = time_slice_max_active - int(num_time_slices_integral/2)
    time_slice_stop = time_slice_start + num_time_slices_integral

    if time_slice_start < 0 or time_slice_stop >= 100:
        continue

    light_field_sequence = event.light_field_sequence_for_isochor_image()
    light_field = np.sum(
        light_field_sequence[time_slice_start:time_slice_stop, :], axis=0)

    # guessing start values
    energy_it = event.simulation_truth.event.corsika_event_header.total_energy_GeV
    shower_altitude_it = 12e3
    core_x_it = event.simulation_truth.event.corsika_event_header.core_position_x_meter()
    core_y_it = event.simulation_truth.event.corsika_event_header.core_position_y_meter()
    momentum = event.simulation_truth.event.corsika_event_header.momentum()
    momentum /= np.linalg.norm(momentum)
    source_cx = -momentum[0]
    source_cy = -momentum[1]

    if np.hypot(core_x_it, core_y_it) > 200:
        print("Core too far out")
        continue

    if energy_it > 24:
        print("Energy too far out")
        continue

    def fit(
        source_cx,
        source_cy,
        core_x_it,
        core_y_it,
        energy_it,
        shower_altitude_it
    ):
        light_field_template = P.lixel_response(
            source_cx,
            source_cy,
            core_x_it,
            core_y_it,
            energy_it,
            shower_altitude_it)
        light_field_template += nsb_in_lixel
        print(
            source_cx,
            source_cy,
            core_x_it,
            core_y_it,
            energy_it,
            shower_altitude_it)
        chi2 = estimate_chi2(
            light_field_measured=light_field,
            light_field_template=light_field_template,
            nsb_in_lixel=nsb_in_lixel)

        inte_template = np.sum(
            light_field_template.reshape(
                lfg.number_pixel,
                lfg.number_paxel),
            axis=1)
        cer_light_field = np.zeros(lfg.number_lixel)
        for lix in event.cherenkov_photons.lixel_ids:
            cer_light_field[lix] += 1
        inte = np.sum(
            cer_light_field.reshape(
                lfg.number_pixel,
                lfg.number_paxel),
            axis=1)

        img_template = pl.image.Image(
            inte_template,
            lfg.pixel_pos_cx,
            lfg.pixel_pos_cy)
        img = pl.image.Image(
            inte,
            lfg.pixel_pos_cx,
            lfg.pixel_pos_cy)
        

        # print("sum img", np.sum(inte), "sum template",  np.sum(inte_template))
        
        fig = plt.figure()
        ax1 = fig.add_axes([0, 0, 1, .5])
        ax2 = fig.add_axes([0, .5, 1, .5])
        pl.plot.image.add_paxel_image_to_ax(img, ax1)
        pl.plot.image.add_paxel_image_to_ax(img_template, ax2)
        fig.savefig("{:012d}.jpg".format(event.number))
        plt.close(fig)
        

        return chi2


    chi2 = fit(
        source_cx,
        source_cy,
        core_x_it,
        core_y_it,
        energy_it,
        shower_altitude_it)
    print(chi2)

    light_field_template = P.lixel_response(
        source_cx,
        source_cy,
        core_x_it,
        core_y_it,
        energy_it,
        shower_altitude_it)

    true_num_cherenkov_photons = event.simulation_truth.detector.number_air_shower_pulses()
    template_num_cherenkov_photons = light_field_template.sum()

    print("------------")
    print("lf true cer:", true_num_cherenkov_photons)
    print("lf templ:", template_num_cherenkov_photons)
    print("ratio lf: ", true_num_cherenkov_photons/template_num_cherenkov_photons)


    """
    m = iminuit.Minuit(
        fit,
        source_cx=source_cx,
        error_source_cx=np.deg2rad(0.1),
        limit_source_cx=(source_cx-d2r(1.5), source_cx+d2r(1.5)),
        fix_source_cx=True,

        source_cy=source_cy,
        error_source_cy=np.deg2rad(0.1),
        limit_source_cy=(source_cy-d2r(1.5), source_cy+d2r(1.5)),
        fix_source_cy=True,

        core_x_it=core_x_it,
        error_core_x_it=10,
        limit_core_x_it=(-120, 120),
        fix_core_x_it=True,

        core_y_it=core_y_it,
        error_core_y_it=10,
        limit_core_y_it=(-120, 120),
        fix_core_y_it=True,

        energy_it=energy_it,
        error_energy_it=0.2,
        limit_energy_it=(0.26, 24.9),
        fix_energy_it=True,

        shower_altitude_it=shower_altitude_it,
        error_shower_altitude_it=1e3,
        limit_shower_altitude_it=(12e3, 15e3),
        fix_shower_altitude_it=True,

        errordef=1)

    a, fa = m.profile(
        'source_cx',
        bins=12,
        bound=(source_cx-d2r(.5), source_cx+d2r(.5)))
    plt.plot(a, fa)
    plt.show()
    """
    # m.migrad()  # run optimiser

