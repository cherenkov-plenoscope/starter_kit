from importlib import resources as importlib_resources
import os


def get_resources_dir():
    return os.path.join(
        importlib_resources.files("cosmic_fluxes"), "pulsars", "resources"
    )
