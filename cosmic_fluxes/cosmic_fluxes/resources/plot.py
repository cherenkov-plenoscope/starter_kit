import cosmic_fluxes
import matplotlib.pyplot as plt

p = cosmic_fluxes.read_cosmic_proton_flux_from_resources()
e = cosmic_fluxes.read_cosmic_electron_positron_flux_from_resources()
he = cosmic_fluxes.read_cosmic_helium_flux_from_resources()

fig = plt.figure(figsize=(16 / 3, 9 / 3), dpi=120 * 3)
ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])

ax.plot(
    p["energy"]["values"],
    p["differential_flux"]["values"],
    color="red",
    label="p",
)

below_2000GeV = np.array(he["energy"]["values"]) < 2000
ax.plot(
    np.array(he["energy"]["values"])[below_2000GeV],
    np.array(he["differential_flux"]["values"])[below_2000GeV],
    color="orange",
    label="He",
)

ax.plot(
    e["energy"]["values"],
    e["differential_flux"]["values"],
    color="blue",
    label="e$^{+}$/e$^{-}$",
)

ax.loglog()
ax.set_xlabel("energy / " + he["energy"]["unit_tex"])
ax.set_ylabel("differential flux / " + he["differential_flux"]["unit_tex"])
ax.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)
ax.spines["top"].set_color("none")
ax.spines["right"].set_color("none")
ax.legend()
fig.savefig("cosmic_rays.png")
