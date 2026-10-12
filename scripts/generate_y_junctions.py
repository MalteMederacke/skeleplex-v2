import os  # noqa: D100

import h5py

from skeleplex.data.y_junctions import (
    generate_y_junction,
    random_parameters_y_junctions,
)

save_path = "./y_junctions/"
os.makedirs(save_path, exist_ok=True)

for i in range(10):
    params = random_parameters_y_junctions()
    skeleton, distance_field = generate_y_junction(*params)
    with h5py.File(os.path.join(save_path, f"y_junction_{i}.h5"), "w") as f:
        f.create_dataset("label", data=skeleton)
        f.create_dataset("raw", data=distance_field)
