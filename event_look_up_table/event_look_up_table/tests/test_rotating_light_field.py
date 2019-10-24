import event_look_up_table as elut
import numpy as np
import tempfile
import os

d2r = np.deg2rad
assert_close = np.testing.assert_almost_equal


def test_rotation_unity():
    NUM_PH = 1
    light_field = np.recarray(NUM_PH, dtype=elut.LIGHT_FIELD_DTYPE)
    light_field.x = [1]
    light_field.y = [-2]
    light_field.cx = [d2r(0.1)]
    light_field.cy = [d2r(-0.3)]

    light_field_t = elut._transform_light_field_to_instrument_frame(
        light_field_in_shower_frame=light_field,
        source_cx_in_instrument_frame=0,
        source_cy_in_instrument_frame=0,
        shower_core_x_in_instrument_frame=0,
        shower_core_y_in_instrument_frame=0)

    for ph in range(light_field.shape[0]):
        assert light_field_t[ph].x == light_field[ph].x
        assert light_field_t[ph].y == light_field[ph].y
        assert light_field_t[ph].cx == light_field[ph].cx
        assert light_field_t[ph].cy == light_field[ph].cy


def test_rotation_cx():
    NUM_PH = 1
    light_field = np.recarray(NUM_PH, dtype=elut.LIGHT_FIELD_DTYPE)
    light_field.x = [0]
    light_field.y = [0]
    light_field.cx = [d2r(0.)]
    light_field.cy = [d2r(0.)]

    light_field_t = elut._transform_light_field_to_instrument_frame(
        light_field_in_shower_frame=light_field,
        source_cx_in_instrument_frame=d2r(45.),
        source_cy_in_instrument_frame=0,
        shower_core_x_in_instrument_frame=0,
        shower_core_y_in_instrument_frame=0)

    for ph in range(light_field.shape[0]):
        assert_close(light_field_t[ph].x, 0.)
        assert_close(light_field_t[ph].y, 0.)
        assert_close(light_field_t[ph].cx, np.sqrt(2.)/2.)
        assert_close(light_field_t[ph].cy, 0.)


def test_rotation_1():
    NUM_PH = 4
    light_field = np.recarray(NUM_PH, dtype=elut.LIGHT_FIELD_DTYPE)
    light_field.x = [1, 1, -1, -1]
    light_field.y = [1, -1, 1, -1]
    light_field.cx = [d2r(0.), d2r(0.), d2r(0.), d2r(0.)]
    light_field.cy = [d2r(0.), d2r(0.), d2r(0.), d2r(0.)]

    light_field_t = elut._transform_light_field_to_instrument_frame(
        light_field_in_shower_frame=light_field,
        source_cx_in_instrument_frame=d2r(45.),
        source_cy_in_instrument_frame=0.,
        shower_core_x_in_instrument_frame=0.,
        shower_core_y_in_instrument_frame=0.)

    lf_expected = np.recarray(NUM_PH, dtype=elut.LIGHT_FIELD_DTYPE)
    lf_expected.x = [np.sqrt(2.),  np.sqrt(2.), -np.sqrt(2.), -np.sqrt(2.)]
    lf_expected.y = [1, -1,  1, -1]
    lf_expected.cx = [
        np.sqrt(2.)/2., np.sqrt(2.)/2., np.sqrt(2.)/2., np.sqrt(2.)/2.]
    lf_expected.cy = [d2r(0.), d2r(0.), d2r(0.), d2r(0.)]

    for ph in range(light_field.shape[0]):
        assert_close(light_field_t[ph].x, lf_expected[ph].x)
        assert_close(light_field_t[ph].y, lf_expected[ph].y)
        assert_close(light_field_t[ph].cx, lf_expected[ph].cx)
        assert_close(light_field_t[ph].cy, lf_expected[ph].cy)


def test_translation_1():
    NUM_PH = 4
    light_field = np.recarray(NUM_PH, dtype=elut.LIGHT_FIELD_DTYPE)
    light_field.x = [1, 1, -1, -1]
    light_field.y = [1, -1, 1, -1]
    light_field.cx = [d2r(0.), d2r(0.), d2r(0.), d2r(0.)]
    light_field.cy = [d2r(0.), d2r(0.), d2r(0.), d2r(0.)]

    """
    observation-level

        o(-1,1)        o(1,1)

                 X-instrument (0,0)

        o(-1,-1)       o(1,-1)
    """

    light_field_t = elut._transform_light_field_to_instrument_frame(
        light_field_in_shower_frame=light_field,
        source_cx_in_instrument_frame=0.,
        source_cy_in_instrument_frame=0.,
        shower_core_x_in_instrument_frame=1.,
        shower_core_y_in_instrument_frame=0.)

    lf_expected = np.recarray(NUM_PH, dtype=elut.LIGHT_FIELD_DTYPE)
    lf_expected.x = [2, 2, 0, 0]
    lf_expected.y = [1, -1, 1, -1]
    lf_expected.cx = [d2r(0.), d2r(0.), d2r(0.), d2r(0.)]
    lf_expected.cy = [d2r(0.), d2r(0.), d2r(0.), d2r(0.)]

    """
    observation-level

                 o( 0,1)        o(2,1)

                 X-instrument (0,0)

                 o( 0,-1)       o(2,-1)
    """

    for ph in range(light_field.shape[0]):
        assert_close(light_field_t[ph].x, lf_expected[ph].x)
        assert_close(light_field_t[ph].y, lf_expected[ph].y)
        assert_close(light_field_t[ph].cx, lf_expected[ph].cx)
        assert_close(light_field_t[ph].cy, lf_expected[ph].cy)
