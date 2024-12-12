
import pymeshlab
import tqdm
import math
import os
from concurrent.futures import ProcessPoolExecutor

from utils import *

# CONDA : remeshing_iso_env

def compute_face_area_stats(mesh):
    """Compute mean, standard deviation, and CV of face areas."""
    face_areas = mesh.face_matrix() # Get the areas of all faces
    mean_area = np.mean(face_areas)
    std_dev = np.std(face_areas)
    cv = std_dev / mean_area if mean_area != 0 else 0
    return mean_area, std_dev, cv

###############################################################################
MododelNet_dir = "/home/pelissier/These-ATER/Papier_international3/Dataset/ModelNet40_centered_scaled"
paths_mesh = read_paths_from_txt('/home/pelissier/These-ATER/Papier_international3/Dataset/paths_files/obj_SMPLER_files_ModelNet40_centered_scaled.txt')

max_iterations = 20  # Safety limit for iterations
cv_threshold = 0.05  # Coefficient of variation threshold for uniformity

results = []  # List to store results
def remeshing_isotropic_mesh(path_mesh):
    try :      
        # Load the mesh
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(path_mesh)
        name_mesh = os.path.basename(path_mesh)
        
        cv_values = []  # List to store CV values
        stop_iteration = None  # To store the iteration where CV repeats
        ## Automatic determine iteration param
        for iter in range(3, max_iterations+1):
            # load the mesh
            ms.load_new_mesh(path_mesh)
            # Apply isotropic remeshing : default param are the same in meshlab
            ms.meshing_isotropic_explicit_remeshing(iterations=iter)
            # Compute face area statistics
            _, _, cv = compute_face_area_stats(ms.current_mesh())
            # Store CV value in the list
            cv_values.append(np.round(cv,3))
            # Check if the CV is below the threshold
            if len(cv_values) >= 3 and (cv_values[-2] == cv_values[-3] == cv_values[-4]):
                stop_iteration = iter-3
                break

        # Save the remeshed result
        ms.save_current_mesh(os.path.join(MododelNet_dir, name_mesh+"_remeshing_iso_iter"+str(stop_iteration)+".obj"))
        results.append(("ok", path_mesh))
        
    except Exception as e: results.append(("pbl", path_mesh))
          
# Main function to parallelize the loop with a progress bar
def process_files_in_parallel(files, max_workers=16):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Use tqdm with executor.map for a progress bar
        results = list(tqdm(executor.map(remeshing_isotropic_mesh, files), total=len(files), desc="Processing files"))
    return results

# Process the files in parallel (adjust the number of workers if necessary)
results = process_files_in_parallel(paths_mesh, max_workers=16)

# Write results to file
with open("/home/pelissier/These-ATER/Papier_international3/Dataset/error_run1_remeshing_iso.txt", "w") as file:
    for name, path in results:
        file.write(f"{name}: {path}\n")