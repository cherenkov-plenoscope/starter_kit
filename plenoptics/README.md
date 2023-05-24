Demonstrate how plenoptics can compensate aberations and spread in time
=======================================================================

The instrument here is a plenoscope made from an imaging mirror and a
consequutive light-field-sensor. The geometry of the imaging mirror, and
the resolution of the light-field-sensor can be varied.

Degrees of Freedom
------------------

- Geometry of mirror
  [```sphere_monolith```, ```davies_cotton```, ```parabola_segmented```].

- Off-axis-angle
  A list of angles in ```cx```.

- Number of photo-sensors in light-field-sensor's cameras
  [1, 3, 9, ...] a list of odd integer numbers specifies the number of
  photo-sensors sampling the diagonal of the mirror.


How it is done
--------------

For each combination of:
- mirror-geometry
- light-field-sensor-resolution
- off-axis-angle

A dedicated plenoscope is simulated. We estimate the geometry of the
light-field that it percieves and estimate its response to a
calibration-source.
This is done individually for each off-axis-angle because of a limitation of
the current geometry of our light-field-sensors. Currently we can only create
light-field-sensors with all cameras in a plane. For small field-of-views
this is fine. But for large field-of-views as we want to explore here it is
better to have a curved surface for the cameras with each camera pointing to
the center of the mirror.
In order to overcome this limitation we only simulate a small
light-field-sensor with a limited field-of-view but move and rotate it
accordingly to make its central cameras point to the center of the mirror.
This is a good approximation for the result one would get when the cameras
would be on a curved, spherical surface.

To do this, we only use one calibration-source with photons coming from zenith.
To simulate off-axis-angles, we rotate the entire plenoscope
(mirror and sensor) accordingly.
