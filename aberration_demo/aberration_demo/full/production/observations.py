from ... import sources


def make_response_to_source(
    source_config, light_field_geometry_path, merlict_config,
):
    if source_config["type"] == "star":
        return sources.star.make_response_to_star(
            star_config=source_config,
            light_field_geometry_path=light_field_geometry_path,
            merlict_config=merlict_config,
        )
    elif source_config["type"] == "mesh":
        return sources.mesh.make_response_to_mesh(
            mesh_config=source_config,
            light_field_geometry_path=light_field_geometry_path,
            merlict_config=merlict_config,
        )
    elif source_config["type"] == "point":
        return sources.point.make_response_to_point(
            point_config=source_config,
            light_field_geometry_path=light_field_geometry_path,
            merlict_config=merlict_config,
        )
    else:
        raise AssertionError("Type of source is not known")
