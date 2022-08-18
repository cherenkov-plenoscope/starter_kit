Detecting gamma-rays with 1 Giga electron Volt
             from ground with the
             Cherenkov-Plenoscope
==============================================

Plenoptic perception of the Cherenkov-light emitted in airshowers offers novel
opportunities. One specific opportunity is the 'gamma-ray-timing-explorer' as
Felix Aharonian coined the term in the 5@5-proposal. From our point-of-view,
the creation of a gamma-ray-timing-explorer is the strongest motivation and
shall be discussed here. Further opportunities, e.g. significantly widening the
field-of-view, shall not be discussed here.

We want to persuade our colleagues that a plenoptic gamma-ray-timing-explorer
is possible. To do this, we have to answer:

- (A) What are the limits of the current methods that prevent us from building
      a gamma-ray-timing-explorer?

- (B) How does plenoptic perception overcome the current limits?

- (C) What would an actual implementation look like, and what resources would
      it take?

- (D) What is the performance of such an implementation?


We plan to cover this in three manuscripts.
We do not discuss the physics-cases in detail, but we mention the advantages of
our proposal whenever they help a general timing-explorer. E.g. we discuss the
cable-mount being intrinsically faster than altitude-azimuth-mounts.

(1) Optics
==========
The optics-manuscript covers the questions (A),(B), and (C).

On question (A): Current methods and their limtis
-------------------------------------------------

First, sattelites.
   - Easy to argue.
   - Sattelites are awesome, but too small. It is unlikely for them to become
        much bigger.

Second, large Cherenkov-telescopes.
   - Easy to argue.
   - Works well to some extent (LST, H.E.S.S.-CT5), but runs into the physical
        limit of a the narrow depth-of-field, and runs into the physical limit
        of the square-cube-law. Can cite CTA and Werner directly for the limits
        of depth-of-field and focussing.
   - I do have text and figures to explain the depth-of-field in detail if
        needed (1 Page). In the past people have asked for a head to head
        comparison of the telescope and the plenoscope w.r.t. depth-of-field.
        Problem is, that requires to draw and explain the thin-lens.
        Werner does not like this.

Third, trigger for an array of Cherenkov-telescopes
   - Difficult to argue.
   - One can not proove that it is an unsolvable challange at the moment.
        I think everyboby agrees that some research is needed to make it work,
        but people differ on the amount of research that is needed.
        This makes it important that we later stress that the plenoscope has no
        technological challanges left to be solved and is ready to go.
   - The only simple, but bad argument, that I can come up with is:
        Well, the array-trigger is out reach because CTA would not build
        telescopes of different sizes if the array-trigger would have been an
        option.
        The argument is that CTA, despite its resources, did not implement an
        array-trigger, although the array-trigger was on the table before CTA
        (STAR-proposal by Jung).
        I do not like to argue this way becasue in the beginning of CTA it was
        decided, for very good reasons, to only use etablished technology
        (no SiPMs, no dual-mirror, ...) what naturally outrules the
        array-trigger.
   - We should cite the implementations closest to an array-trigger
         (MAGIC-topo-trigger, VERITAS-topo-trigger-design-study).

The big achivement of the Cherenkov-plenoscope is that it
   - turns the tables on the physical limit of the narrow depth-of-field
       Recorded light-field can be projected onto images with arbitrary focus.
   - defers the physical limit of the square-cube-law long enough to reach 1GeV
       Can tolerate deformations and misalignments. The structure is
       allowed to flex.
   - works around the technological challange of beam-forming in array-trigger
       Photo-sensors sampling same directions are mechanically close and do not
       need variable delays.
   - widens and flattens the field-of-view by reducing aberrations.

On question (B): How
--------------------
This is all about knowing the geometry of the beams of the photo-sensors which
sample the light-field. Refocusing, compensating aberrations and deformations
is all included in this knowledge. It does help to show the thin-lens
setup/equation in this context.

We must point out that knowing the light-field-geometry of a plenoscope allows
us to abstract away from all the optics quirks and features and directly get
the observables of the photons in great detail.
[position of impact (x, y), direction (c_x, c_y), time of impact (t)]
This is why I once tried to make the story based on the observables.

The image formation, the refocussing, the compensation of deformations and
aberrations, the timing corrections, are all covered by the same formalism.

Any quali- and quantitative improvement is demonstrated later on the specific
implementation of Portal.

We might mention that the concept of knowing an instrument's
light-field-geometry is very general and applies to any optics with
photo-sensors. It overcomes current limitations that occur when identifying
photo-sensors with pixels in an image what prevents you from compensating even
simple distortions.

On question (C): What implementation
------------------------------------
Here we introduce the 71m Portal Cherenkov-plenoscope.
To stress that no technological challenges are left when building Portal we
have to describe its intendet simplicity.
- Use facets not larger than 2m^2 as its done in the LST.
- All optical surfaces are spherical
- All optical surfaces are mass-producible.
    Only two repeating primitives. The mirror-facet and the lens.
- The lenses are just made out of silica-glass and do not need surfaces better
    than windows.
- The cable-robot is well researched and even transports human passengers.
- The hardware of the cable-robot can be taken from harbour-cranes.
- The towers of the mount common in industry and do not need novel engineering.
- The space-truss for the mirror is the same technology as in the LST.
- The trigger is just a sum-trigger.
    Just like in a/some telescope the trigger does nothing but summing up
    photo-sensors and comparing the sum against a threshold.
    Yes, there are more sums but they do not interfer or communicate.
    Same technology.

Cameras (small)
...............
Since the small cameras with their lenses are a novelty, we might show their
performance. I got point-spread-functions for those. But it gets very techincal
very quickly like with the reflecting walls inside the cameras.
The important part here for the reader is, that these lenses do not need to be
very good at imaging. We only optimize for good fill-factor and transmission
to keep the Cherenkov-light.

Trigger
.......
We have to describe the possibility of the trigger to act on refocussed images.
We must explain that the effort of creating two images at differnt foci is
exceptable and allows to veto airshowers emitting light at certain altitudes.

Cable-robot
...........
The reader needs to understand that
   - It is easy to build
        Towers are common in industry. No novel engineering.
        No giant joints.
        Small repeating parts -> Easy to mass-produce.
   - It is intrinsically faster than an altitude-azimuth-mount
        No near-zenith singularity.
   - It is easy to service the light-field-sensor
        Can park independent of mirror on pedestal.
   - It is rather safe during the day.
        Light-field-sensor does not park near the focus during daytime.

Costs
.....
We do have a cost-estimate
   - Electronics and photo-sensors (myself)
   - Mechanical structure and optics (civil engineers, myself)
   - Infrastructure, enginieering and manpower for the errection at a harsh and
     remote site (based on E.S.O. experience).

I was thinking about the cost of operation, especially electric power.
Afterall it is 2022. Maybe we have to give an estimate for a solar/wind
power-plant with combined storage-unit at the site.
A back-on-the-envelope-estimate suggests that this raises Portal's cost by 5%.

(2) Background
==============
The background-manuscript covers a part of question (D).
This is about the background induced by cosmic-rays below 10GeV.
Going below 10GeV brings qualitative changes which can not be estimated or
extrapolated with current methods.
Going below 10GeV is a novelty worth documenting.
The qualitative change is the strong deflection of charged particles in the
shower what makes the shower bend significantly.
This bending can exceed 90 degrees for leptons and thus creates novel
challenges. We describe a novel algorithm to estimate the instrument-response
of a general Cherenkov-instrument with a trigger-threshold below 10GeV.
We have to answer:

- (2.algo.a) What is the qualitative difference when estimating the
        instrument-response below 10GeV?
    The shower is bending.
    The direction of the cosmic particle is no longer the meadian direction
    of the Cherenkov-light.
    Show images of light-pools on ground for leptons (huge spread and scatter).
    Show how the direction of cosmic particle needs to change with energy to
    always get Cherenkov-light from certain direction.
    Mention the existing of an in-atmosphere-cutoff for leptons at some sites
    which is on top of the geomagnetic cutoff.

- (2.algo.b) Why can it not be done efficiently with our current algorithms?
        (Bernlohrs iact.c)
    The current algorithms work with the scatter-radius and assume a uniform
        coverage of Cherenkov-light on ground.
    The scatter-radii needed to respect the bending results in a very
        inefficient population.
    I think we should restrict the discussion to the algorithm and not the
        current implementations. Because otherwise:
            Also bug-fixes in CORSIKA for electrons below approx. 2GeV...
            Also CORSIKA's input-structure does not allow to correltate
                direction with energy...
            Also CORSIKA's output-formats have issues with the amount of
                Cherenkov-photons in TeV showers hitting the plenoscope.

- (2.algo.c) What would an efficient algorithm look like?
    - You must give up on scatter-radius.
    - You need to know what direction a cosmic particle must have to get
      Cherenkov-light from a certain direction.
    - You must not pick a scatter-position at random, but only from a pool
      of scatter-positions that have a reasonable chance to trigger.
    - Show Werner's estimate for uncertainty.
    - Show that our algorithm reproduces the existing algorithm for
      energies above 10GeV.

Since earth's magnetic-field plays an important role we can discuss differences
between popular sites:
   - The horizontal component of earth magnetic-field probably effects
     the reconstruction of the direction of gamma-rays.
   - Show spread of gamma-ray-directions for different sites
     (Namibia, Chile, La Palma)
   - Show density of Cherenkov-light on ground for differnet sites.

For Cherenkov-method we care about the flux of airshowers and not particles.
We have to point out the difference of the flux of particles on the edge of
earth's atmosphere and the flux of airshowers induced by these particles.
This is where the geomagnetic cutoff strikes. This is where we have to fallback
on estimates for the flux of airshowers induced by terrestrial/secondary-rays.


(3) Performance
===============
The performance-manuscript covers the remaining part of question (D).

If we have a detailed listing of physics-cases I think they should be listed
here. Except for gravitational counterparts this is basically the same
motivation as for 5@5.

Gamma-ray-bursts
   - High probability to see GRB by chance because of high observable distances.

Counterparts to gravitational mergers
   - Probably enough warning-time to point the plenoscope

Timing-array for gravitational waves using pulsars
   - Way too fancy :)

Active-galactic-nuclei
   - Probably the go-to-target when there are no alerts.
   - Restricting size of emission region.
   - Testing Lorentz invariance? 1GeV vs. 1TeV?

Fast-radio-bursts
   - Counterparts? Need to check rate of random occurance in field-of-view.

Flares of the Crab-nebula ...

Current state
-------------
The reader must understand that this is probably not the best reconstruction
and analysis possible. Afterall the reconstruction of energy and direction
might improve in the future. We shoud say that there are intrinsic limits due
to limited statistics of particles in the airshowers and that therefore energy
and direction are unlikely to become ever as good as at 1TeV.
Especially with g/h-seperation. Here we lack far behind of what the LST can do
at 25GeV.

Setup and environment
---------------------

Site
   - Location and altitude
         Explain why Chile atacama-desert and Namibia Gamsberg are good but
         differnet choices. Reference site-comparison in background-manuscript.
   - Atmosphere properties
   - Earth's magnetic-field

Night-Sky
   - Show differential flux of light

Air-showers induced by cosmic-rays
   - We include H, He, e^+/e^-
   - Show differential flux for air-showers (not for particles), reference our
        background-manuscript.

Instrument
   - Refernece Portal from our optics-manuscript.
   - Sometimes people also list here all the non-geometry aspects of the
        instrument such es reflectivity of mirrors or detection efficiency of
        photo-sensors. Not sure whether this should go here or to the
        introduction of Portal.
   - Performance to reconstruct the arrival-time of single photon.
   - Explain how the trigger's logic is set-up
        Maybe explain this already in the optics-manuscript?
        Two images on two depths to veto on certain depths.
        Favors gamma-rays over hadrons.
   - Any saturation, hick-ups, dead-times? Is the trigger free of dead-time?

Performance figures
-------------------

Ratescan
   - Shows that the trigger-rates are possible with existing technology i.e.
        < 100k s^{-1}.
   - Shows the fraction of random triggers on light from the night-sky.

Gamma-hadron seperation
   - Probably we will not use g/h-seperation here. In this case it is important
         to point this out.

Direction resolution vs. true-energy
   - 68% containmeint
   - Compare CTA-south/Fermi-LAT

Energy resolution vs. true-energ
   - 68% containment
   - Compare CTA-south/Fermi-LAT

Differential sensitvity for observation time of 180s (3min)
   - Compare CTA-south/Fermi-LAT
   - A novel flavor of differential sensitivity to respect poor
         energy-resolution. We might try explain this flavor very briefly.
         But I am afraid that when we mention this, we will have to explain
         it in all its detail. This is a manuscript in its own. Not impossible,
         I already presented a comparism and formalism on our MPIK-retreat.
         But maybe we just do not want to even mention this here at all.
         Just say: 'differential sensitivity'.
   - Systematic uncertainties.
         I am afraid that when we argue that we can go below 1e-2, we will
         probably have to explain a lot despite our argument of having more
         statistics from higher rates.

Conclusion
----------
This is the first proposal for 1GeV at collection-areas > 10^3m^2.
This is the the gamma-ray-timing-explorer.
Now it is only about the costs. And the costs (220e6) are only a fraction of
a sattelite.

Strong yet simple examples:
For a given flux, e.g. GRBs of the past. We can compare how fine the sampling
of time would have been with CTA-south, Fermi-LAT, and Portal.

