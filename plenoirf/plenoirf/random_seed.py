import corsika_primary_wrapper as cpw

STRUCTURE = cpw.random_seed.CorsikaRandomSeed(
    NUM_DIGITS_RUN_ID=6,
    NUM_DIGITS_AIRSHOWER_ID=3
)
