import os  # noqa: D100

import h5py

from skeleplex.data.fractal_trees import (
    generate_random_parameters_for_fractal_tree,
    generate_synthetic_fractal_tree,
)

save_path = "./fractal_trees/"
os.makedirs(save_path, exist_ok=True)

for i in range(10):
    params = generate_random_parameters_for_fractal_tree(seed=i)
    skeleton, distance_field = generate_synthetic_fractal_tree(*params)
    with h5py.File(os.path.join(save_path, f"fractal_tree_{i}.h5"), "w") as f:
        f.create_dataset("label", data=skeleton)
        f.create_dataset("raw", data=distance_field)
