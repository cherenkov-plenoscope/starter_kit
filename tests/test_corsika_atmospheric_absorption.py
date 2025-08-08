import os
import subprocess as sp
import tempfile
import numpy as np
import corsika_primary_wrapper as cpw


MERLICT_EVENTIO_CONVERTER = os.path.join(
    "..", "build", "merlict", "merlict-eventio-converter"
)
CORSIKA_VANILLA = os.path.join(
    "..", "build", "corsika", "original", "corsika-7.56", "run", "sth"
)


def steering_card(obslevel_in_cm, atmospheric_absorbtion):
    out = ""
    out += "RUNNR   1\n"
    out += "EVTNR   1\n"
    out += "NSHOW   1\n"
    out += "PRMPAR  1\n"
    out += "ESLOPE  0.\n"
    out += "ERANGE  1. 1.\n"
    out += "THETAP  0. 0.\n"
    out += "PHIP    0. 360.\n"
    out += "SEED 1 0 0\n"
    out += "SEED 2 0 0\n"
    out += "SEED 3 0 0\n"
    out += "SEED 4 0 0\n"
    out += "OBSLEV  {:f}\n".format(obslevel_in_cm)
    out += "FIXCHI  0.\n"
    out += "MAGNET 1e-99 1e-99\n"
    out += "ELMFLG  T   T\n"
    out += "MAXPRT  1\n"
    out += "PAROUT  F F\n"
    out += "TELESCOPE  0. 0. 0. 5000e2 # instrument radius\n"
    out += "ATMOSPHERE 26 T # Chile, Paranal, ESO site\n"
    out += "CWAVLG 250 700\n"
    out += "CSCAT 1 0 0\n"
    out += "CERQEF F {:s} F # pde, atmo, mirror\n".format(
        "T" if atmospheric_absorbtion else "F"
    )
    out += "CERSIZ 1\n"
    out += "CERFIL F\n"
    out += "TSTART T\n"
    out += "EXIT\n"
    return out


def evaluate_steering_card(steering_card):
    with tempfile.TemporaryDirectory(prefix="test_corsika_") as tmp:
        eventio_path = os.path.join(tmp, "shower.eventio")
        simpleio_path = os.path.join(tmp, "shower.simpleio")
        cpw.corsika_vanilla(
            corsika_path=CORSIKA_VANILLA,
            steering_card=steering_card,
            output_path=eventio_path,
            stdout_path=eventio_path + ".stdout",
            stderr_path=eventio_path + ".stderr",
        )
        cpw.testing.eventio_to_simpleio(
            merlict_eventio_converter=MERLICT_EVENTIO_CONVERTER,
            eventio_path=eventio_path,
            simpleio_path=simpleio_path,
        )
        run = cpw.testing.SimpleIoRun(path=simpleio_path)
        evth, cpb = next(run)

        return {
            "mean_prob_to_reach_obslevel": float(
                np.mean(cpb.probability_to_reach_observation_level)
            ),
            "num_bunches": int(
                cpb.probability_to_reach_observation_level.shape[0]
            ),
            "median_wavelength": float(np.median(cpb.wavelength)),
        }


def test_corsika_atmospheric_absorbtion():
    observation_levels = np.linspace(1.0e3, 6.0e3, 4)
    i = {}
    i["ATMABS"] = 0
    i["OBSLEV"] = 1
    i["MNPROB"] = 2
    i["NUMBUN"] = 3
    i["MEDWVL"] = 4
    results = np.zeros(shape=(4 * 2, 5))
    idx = 0
    for atmabs in [True, False]:
        for observation_level in observation_levels:
            r = evaluate_steering_card(
                steering_card(
                    obslevel_in_cm=observation_level * 1e2,
                    atmospheric_absorbtion=atmabs,
                )
            )
            results[idx, i["ATMABS"]] = 1.0 if atmabs else 0.0
            results[idx, i["OBSLEV"]] = observation_level
            results[idx, i["MNPROB"]] = r["mean_prob_to_reach_obslevel"]
            results[idx, i["NUMBUN"]] = r["num_bunches"]
            results[idx, i["MEDWVL"]] = r["median_wavelength"]
            idx += 1

    atmabs_T = results[:, i["ATMABS"]] == 1.0
    atmabs_F = results[:, i["ATMABS"]] == 0.0

    fit_atmabs_T = np.polyfit(
        x=results[atmabs_T, i["OBSLEV"]][0:3],
        y=results[atmabs_T, i["NUMBUN"]][0:3],
        deg=1,
    )

    fit_atmabs_F = np.polyfit(
        x=results[atmabs_F, i["OBSLEV"]][0:3],
        y=results[atmabs_F, i["NUMBUN"]][0:3],
        deg=1,
    )

    np.testing.assert_approx_equal(
        actual=fit_atmabs_T[0], desired=3.8, significant=1
    )

    np.testing.assert_approx_equal(
        actual=fit_atmabs_F[0], desired=0.0, significant=2
    )

    fit_wvl_atmabs_T = np.polyfit(
        x=results[atmabs_T, i["OBSLEV"]],
        y=results[atmabs_T, i["MEDWVL"]] * -1e9,
        deg=1,
    )

    fit_wvl_atmabs_F = np.polyfit(
        x=results[atmabs_F, i["OBSLEV"]],
        y=results[atmabs_F, i["MEDWVL"]] * -1e9,
        deg=1,
    )

    np.testing.assert_approx_equal(
        actual=fit_wvl_atmabs_T[0], desired=-0.00586, significant=3
    )

    np.testing.assert_approx_equal(
        actual=fit_wvl_atmabs_F[0], desired=-0.000017, significant=1
    )

    """
    SAVE RESULTS
    """
    out_dir = "test_atmospheric_absorbtion"
    os.makedirs(out_dir, exist_ok=True)

    header = "test CORSIKA atmospheric absorbtion\n"
    for key in i:
        header += "{:d} {:s}\n".format(i[key], key)

    np.savetxt(
        os.path.join(out_dir, "results.csv"),
        results,
        delimiter=",",
        header=header,
    )

    """
    fig = plt.figure()
    ax = fig.add_axes([.15, .15, .8, .8])
    ax.plot(
        results[atmabs_T, i["OBSLEV"]],
        results[atmabs_T, i["NUMBUN"]],
        'bx-',
        label='atmabs True')
    ax.plot(
        results[atmabs_F, i["OBSLEV"]],
        results[atmabs_F, i["NUMBUN"]],
        'rx-',
        label='atmabs False')
    ax.set_xlabel('observation-level / m')
    ax.set_ylabel('num bunches / 1')
    ax.legend()
    fig.savefig(os.path.join(out_dir, 'num_bunches_vs_obslevels.png'))


    fig = plt.figure()
    ax = fig.add_axes([.15, .15, .8, .8])
    ax.plot(
        results[atmabs_T, i["OBSLEV"]],
        results[atmabs_T, i["MNPROB"]],
        'bx-',
        label='atmabs True')
    ax.plot(
        results[atmabs_F, i["OBSLEV"]],
        results[atmabs_F, i["MNPROB"]],
        'rx-',
        label='atmabs False')
    ax.set_xlabel('observation-level / m')
    ax.set_ylabel('mean prob. to reach obslevel / 1')
    ax.legend()
    fig.savefig(
        os.path.join(out_dir, 'nmean_prob_to_reach_obslevel_vs_obslevels.png'))


    fig = plt.figure()
    ax = fig.add_axes([.15, .15, .8, .8])
    ax.plot(
        results[atmabs_T, i["OBSLEV"]],
        results[atmabs_T, i["MEDWVL"]]*-1e9,
        'bx-',
        label='atmabs True')
    ax.plot(
        results[atmabs_F, i["OBSLEV"]],
        results[atmabs_F, i["MEDWVL"]]*-1e9,
        'rx-',
        label='atmabs False')
    ax.set_xlabel('observation-level / m')
    ax.set_ylabel('median wavelength / nm')
    ax.legend()
    fig.savefig(os.path.join(out_dir, 'median_wavelength_vs_obslevels.png'))
    """
