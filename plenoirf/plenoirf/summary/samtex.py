import dominate

HEADER_FONT_FAMILY = "calibri"

TEXT_ALIGNS = [
    "center",
    "left",
    "right",
    "justify",
]


def h(text, level=1, font_family="calibri", text_align="left"):
    map_level = {
        1: dominate.tags.h1,
        2: dominate.tags.h2,
        3: dominate.tags.h3,
        4: dominate.tags.h4,
        5: dominate.tags.h5,
        6: dominate.tags.h6,
    }
    _h = map_level[level](text)
    _h["style"] = "font-family:{:s};".format(
        font_family
    ) + "text-align:{:s}".format(text_align)
    return _h


def p(
    text,
    font_size=100,
    font_family="calibri",
    text_align="left",
    line_height=100,
    width=640,
):
    _p = dominate.tags.p(text)
    _p["style"] = (
        "font-size:{:.1f}%;".format(float(font_size))
        + "font-family:{:s};".format(font_family)
        + "text-align:{:s};".format(text_align)
        + "width:{:.1f}px;".format(float(width))
        + 'line-height:{:.1f}%"'.format(float(line_height))
    )
    return _p


def code(
    text,
    font_family="courier",
    font_size=100,
    line_height=100,
    margin_px=0,
    padding_px=0,
    text_align="left",
):
    _pre = dominate.tags.pre()
    _pre["style"] = (
        "font-size:{:.1f}%;".format(font_size)
        + "font-family:{:s};".format(font_family)
        + "line-height:{:.1f}%;".format(line_height)
        + "text-align:{:s};".format(text_align)
        + 'margin:{:.1f}px;padding:{:.1f}px">'.format(margin_px, padding_px)
    )
    _code = dominate.tags.code(text)
    _pre.add(_code)
    return _pre


def img(src, width_px):
    _img = dominate.tags.img()
    _img["src"] = src
    _img["style"] = "width:{:.1f}px".format(width_px)
    return _img


def table(matrix, width_px):
    _table = dominate.tags.table()
    _table["style"] = "width:{:.1f}px".format(float(width_px))

    for row in matrix:
        _row = dominate.tags.tr()
        for col in row:
            _cell = dominate.tags.td(col)
            _row.add(_cell)
        _table.add(_row)
    return _table
