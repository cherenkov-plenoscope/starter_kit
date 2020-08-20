import numpy as np
import sparse_numeric_table as spt
import magnetic_deflection as mdfl


def _divide_silent(numerator, denominator, default):
    valid = denominator != 0
    division = np.ones(shape=numerator.shape) * default
    division[valid] = numerator[valid] / denominator[valid]
    return division


def effective_quantity_for_grid(
    energy_bin_edges_GeV,
    energy_GeV,
    mask_detected,
    quantity_scatter,
    num_grid_cells_above_lose_threshold,
    total_num_grid_cells,
):
    """
    Returns the effective quantity and its uncertainty.

    Parameters
    ----------
    energy_bin_edges_GeV            Array of energy-bin-edges in GeV

    energy_GeV                      Array(num. thrown airshower)
                                    The energy of each airshower.

    mask_detected                   Array(num. thrown airshower)
                                    A flag/weight for each airshower marking
                                    its detection.

    quantity_scatter                Array(num. thrown airshower)
                                    The scatter-quantity for each airshower.
                                    This is area/m^2 for point like sources, or
                                    acceptance/m^2 sr for diffuse sources.

    num_grid_cells_above_lose_threshold     Array(num. thrown airshower)
                                            Num. of grid cells passing the lose
                                            threshold of the grid for each
                                            airshower.

    total_num_grid_cells            Array(num. thrown airshower)
                                    The total number of grid-cells thrown for
                                    each airshower.


    Formula
    -------
    Q_effective &=& /frac{ Q_detected }{ C_thrown }

    Q_detected &=& /sum_m^M { f_{detected,m} N_{S,m} Q_{scatter,m} }

    C_thrown &=& /sum_m^M { f_{thrown,m} N_G }

    /frac{
        /Delta_Q_effective
    }{
        Q_effective
    } &=& /frac{
        /sqrt{ /sum_m^M { f_{detected,m} N_{S,m}^2 } }
    }{
        /sum_m^M { f_{detected,m} N_{S,m} }
    }

    Variables
    ---------
    N_G             Num. bins in grid.

    N_{S,m}         Num. bins in grid with cherenkov-photons
                    above losethreshold for m-th air-shower.

    Q_{scatter,m}   Scatter-quantity of m-th air-shower.
                    Scatter-area times scatter-solid-angle

    f_{thrown,m}    Flag marking that m-th air-shower is a valid thrown.

    f_{detected,m}  Flag marking that m-th air-shower is a valid detection.

    """

    quantity_detected = np.histogram(
        energy_GeV,
        bins=energy_bin_edges_GeV,
        weights=(
            mask_detected
            * num_grid_cells_above_lose_threshold
            * quantity_scatter
        ),
    )[0]

    count_thrown = np.histogram(
        energy_GeV, weights=total_num_grid_cells, bins=energy_bin_edges_GeV
    )[0]

    effective_quantity = _divide_silent(
        numerator=quantity_detected, denominator=count_thrown, default=0.0
    )

    # uncertainty
    # according to Werner EffAreaComment.pdf 2020-03-21 17:35

    A_square = np.histogram(
        energy_GeV,
        bins=energy_bin_edges_GeV,
        weights=(mask_detected * num_grid_cells_above_lose_threshold ** 2),
    )[0]

    A = np.histogram(
        energy_GeV,
        bins=energy_bin_edges_GeV,
        weights=(mask_detected * num_grid_cells_above_lose_threshold),
    )[0]

    effective_quantity_uncertainty = _divide_silent(
        numerator=np.sqrt(A_square), denominator=A, default=np.nan
    )

    return effective_quantity, effective_quantity_uncertainty
