import os
import trimesh
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import time
import random

from utils import *
from fonction_PtofView import *

# CONDA ENV : blender_env

# Record the start time
start_time = time.time()

# Paths of mesh
mesh_paths = read_paths_from_txt('/home/pelissier/These-ATER/Papier_international3/Dataset/paths_files/obj_SMPLER_files_ModelNet40_centered_scaled.txt')
random.shuffle(mesh_paths)

def process_on_mesh(path_mesh):
    try:
        # Load the mesh
        mesh = trimesh.load(path_mesh)
        
        directory, filename = os.path.split(path_mesh)
        name = filename.split('.')[0] # ex:  piano_0001_centered_scaled
        path_output = os.path.join(directory, name + '_remeshing2.obj')

        # # Check if the mesh is watertight (optional, but helps avoid issues)
        # if not mesh.is_watertight:
        #     print("Warning: The mesh is not watertight.")

        # Subdivide the mesh to increase face regularity (loop subdivision)
        subdivided_mesh = mesh.subdivide()
        subdivided_mesh = subdivided_mesh.subdivide()

        # Save the subdivided mesh
        subdivided_mesh.export(path_output)

        return ('ok', path_mesh)
    
    except Exception as e: 
        return ('pbl', path_mesh)

# Main function to parallelize the loop with a progress bar
def process_files_in_parallel(files, max_workers=16):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Use tqdm with executor.map for a progress bar
        results = list(tqdm(executor.map(process_on_mesh, files), total=len(files), desc="Processing files"))
    return results

# Process the files in parallel (adjust the number of workers if necessary)
results = process_files_in_parallel(mesh_paths, max_workers=16)

# Record the end time
end_time = time.time()
# Calculate and print the running time
running_time = end_time - start_time
print(f"Running time: {running_time:.2f} seconds for {len(mesh_paths)} files")

# Write results to file
with open("/home/pelissier/These-ATER/Papier_international3/Dataset/error_run2_remeshing.txt", "w") as file:
    for name, path in results:
        file.write(f"{name}: {path}\n")
            
            

