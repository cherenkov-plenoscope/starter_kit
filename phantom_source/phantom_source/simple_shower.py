import numpy as np


def num_particles_next_gen(max_particles, prng):
    return int(np.ceil(prng.uniform(0, max_particles)**2)**(1/2))


def append_random_edge(mesh, start_vkey, radius_xy, min_depth, max_particles, N, prng):
    start_pos = mesh["vertices"][start_vkey]
    print(start_vkey, start_pos)
    if start_pos[2] > min_depth:

        num_new = num_particles_next_gen(max_particles=max_particles, prng=prng)
        for n in range(num_new):
            dz = start_pos[2] * prng.uniform(0.125, 0.25)
            end_pos = np.array(
                [
                    start_pos[0] + prng.uniform(-radius_xy, radius_xy)*0.001*dz,
                    start_pos[1] + prng.uniform(-radius_xy, radius_xy)*0.001*dz,
                    start_pos[2] - dz,
                ]
            )
            end_vkey = str(start_vkey) + ".{:d}".format(n)

            mesh["vertices"][end_vkey] = end_pos
            mesh["edges"].append((start_vkey, end_vkey, N))
            append_random_edge(
                mesh=mesh,
                start_vkey=end_vkey,
                radius_xy=radius_xy,
                min_depth=min_depth,
                max_particles=max_particles,
                N=N,
                prng=prng,
            )
