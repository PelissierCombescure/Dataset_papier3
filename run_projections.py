import pandas as pd
import os
import math
import trimesh
from scipy.spatial.transform import Rotation
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import time
import random

from utils import *
from fonction_PtofView import *

# CONDA ENV : blender_env

# Record the start time
start_time = time.time()

# Load the dataset of cameras
circular_config_12_elevation_30 = pd.read_csv("/home/pelissier/These-ATER/Papier_international3/Dataset/Rendu/ModelNet40/circular_config_12_elevation_30.csv")

# Root of outputs folder
dir_output = '/home/pelissier/These-ATER/Papier_international3/Dataset/Rendu/ModelNet40/my_circular_12_elevation_30_remeshing'

# Paths of mesh
mesh_paths = read_paths_from_txt('/home/pelissier/These-ATER/Papier_international3/Dataset/paths_files/obj_files_ModelNet40_centered_scaled_remeshing1-1000.txt')
print(f"Number of files: {len(mesh_paths)}")
print(mesh_paths[0])
print("--------\n")


def process_on_mesh(path_mesh):
    try :
        ## Récupération du nom du fichier
        directory, filename = os.path.split(path_mesh)
        name = filename.split('.')[0] # ex:  piano_0001_centered_scaled
        directory_output = directory.replace('/home/pelissier/These-ATER/Papier_international3/Dataset/ModelNet40_centered_scaled', dir_output)
  
        ## Récupération des coordonnées des sommets, des faces et des normales
        mesh = trimesh.load_mesh(path_mesh)
        array_coords = np.array(mesh.vertices); nb_vertices = len(array_coords) 
        array_faces = np.array(mesh.faces); nb_faces = len(array_faces)
        ## Utiliser le dictionnaire adjacent_faces obtenu précédemment
        adjacent_faces = find_adjacent_faces(array_faces)
        idx_faces_per_idx_vert = find_faces_for_vertices(adjacent_faces)
        array_normals = np.array(mesh.vertex_normals)
        array_normals_face = np.array(mesh.face_normals)
        centroid = np.around(get_centroid(array_faces, array_coords),2)
        diagonal = compute_bounding_box_diagonal(np.array(mesh.bounding_box.vertices)); 

        ## Cam k
        for k in range(1,circular_config_12_elevation_30.shape[0]):
            data_blender_cam = circular_config_12_elevation_30.loc[k].to_dict()
            # Data de la projection
            focale = 300*diagonal; W_image = 400; H_image = 400
            # Positon 3D de la caméra 
            cam = np.array([data_blender_cam['LocationX'], data_blender_cam['LocationY'], data_blender_cam['LocationZ']])
            # POI : centre du mesh
            lookat = centroid
            # vecteur de direction 
            d = lookat - cam
            # vecteur up
            # ATTENTION il y a -1 car un a l'axe y vers le haut (?)
            up = np.array([0, 0, -1])
            # Projection : Monde --> Caméra
            #Matrice de passage entre le repère Monde --> repère Caméra # j'ai verifié : 
            # np.around(np.dot(WorldToCamera,np.array(list(cam)+[1])),5) = [0,0,0,1]
            WorldToCamera = get_worldTocamera_matrix(cam, lookat, up)
            # coordonnées des sommets dans le repère caméra
            array_coords_camera = monde_to_camera(WorldToCamera, array_coords) #[vertices, 1]
            # Projection des normals dans le repère caméra ie on n'applique que la rotation et nn la translation 
            R = WorldToCamera[:3,:3]
            array_normals_camera = np.transpose(np.dot(R, np.transpose(array_normals)))
            array_normals_camera_norm = np.array(normalisation(array_normals_camera[:,:3]))
            ## Pareil : projection + normalisation des normales des faces 
            array_normals_face_camera = np.transpose(np.dot(R, np.transpose(array_normals_face)))
            array_normals_face_camera_norm = np.array(normalisation(array_normals_face_camera[:,:3]))
            ### Projection: Caméra -> image 2D
            ## On recupère les indices des sommets visibles
            array_pixels = camera_to_image(focale, W_image, H_image, array_coords_camera)
            dephtMap, _, _ = get_visible_vertices(W_image, H_image, array_faces, array_pixels, array_coords_camera, array_normals_camera_norm)
            
            ####################################################################################################
            ### FACES VISIBLES - Back face culling
            ## Centre de chaque face : coord des centre de chaque face 
            arr_centre_faces = np.array([compute_face_centre(f, array_coords_camera[:,:3]) for f in array_faces])
            ## On veut l'angle entre la normale de la face et le vecteur [cam, face_centre] dans le repère caméra
            ## comme on est dans le repère caméra, les coord 3D de la caméra sont (0,0,0) vu que par principe le repère caméra est centré en la caméra.
            ## Donc : cam - face_centre = - face_centre
            rayons = -1*arr_centre_faces.copy()
            ## normalisation des rayons sortants
            ## vecteur avec la 1/norme de chaque ligne == chaque centre
            vect_norm = 1/np.linalg.norm(rayons, axis=1)
            ## on reshape en nb_facesx1
            vect_norm = np.reshape(vect_norm, (rayons.shape[0],1))
            ## on repete le vecteur sur 3colonnes car on a des coords 3D / face
            matrix_norm = np.matlib.repmat(vect_norm, 1, 3)
            ## produit terme a terme 
            rayons_norm = rayons*matrix_norm
            ## normal de face dans le repère camera qui est déjà normalisé
            normales_face_norm = array_normals_face_camera_norm.copy()
            ## produit scalaire entre rayon et normale faces
            ## produit scalaire = |a|*|b|*cos(a,b) or a et b sont normalisé, donc ici on a : cos(a,b)
            arr_cos = np.sum(rayons_norm*normales_face_norm, axis=1)
            ## face visible si cos >= 0
            epsilon_cos = 10e-5
            idx_faces_visibles = np.where(arr_cos >= epsilon_cos)[0]

            ############################################################
            ### FACES VISIBLES - filtrage occultations
            vrai_idx_faces_visibles = filtrage_faces_occultees2(arr_centre_faces, idx_faces_visibles, dephtMap, focale, W_image, H_image, epsilon_z=10e-2)
            
            ############################################################
            ### SOMMETS VISIBLES
            idx_vert_visible = find_vertices_with_visible_faces(idx_faces_per_idx_vert, np.array(vrai_idx_faces_visibles))

            ############################################################
            ### SOMMETS VISIBLES - filtrage on garde vraiment ceux qui ont un cos >0
            rayons = -1*array_coords_camera[list(idx_vert_visible)][:,:3].copy()
            ## normalisation des rayons sortants
            # vecteur avec la 1/norme de chaque ligne == chaque vertex
            vect_norm = 1/np.linalg.norm(rayons, axis=1)
            # on reshape en nb_sommetx1
            vect_norm = np.reshape(vect_norm, (rayons.shape[0],1))
            # on repete le vecteur sur 3colonnes car on a des coords 3D /vertice
            matrix_norm = np.matlib.repmat(vect_norm, 1, 3)
            # produit terme a terme 
            rayons_norm = rayons*matrix_norm
            ## normal de pt_index dans le repère camera qui est déjà normalisé
            normales_norm = array_normals_camera_norm[list(idx_vert_visible)].copy()
            # produit scalaire entre vertice et sa normale
            # produit scalaire = |a|*|b|*cos(a,b) or a et b sont normalisé, donc ici on a : cos(a,b)
            arr_cos_vert = np.sum(rayons_norm*normales_norm, axis=1)
            vrai_idx_vert_visible = np.array(list(idx_vert_visible))[np.where(arr_cos_vert>0)]
            
            
            ######################################################################################################
            ## Surface Totale 3D
            surface3D = 0
            for idx_f in range(array_faces.shape[0]):
                face = array_faces[idx_f, :]
                ## coord 3D des 3 sommets la face visible courante dans le rep camera
                sommet0 = array_coords[face[0],:]
                sommet1 = array_coords[face[1],:]
                sommet2 = array_coords[face[2],:]
                ## surface 3D
                surface3D = surface3D + calculer_aire_triangle_3D(sommet0, sommet1, sommet2)
            
            ## Surface visible de la projection courante 
            arr_coords_cam = array_coords_camera[:,:3]
            surface3D_visible = 0
            for idx_f in range(len(vrai_idx_faces_visibles)):
                face = array_faces[idx_f, :]
                ## coord 3D des 3 sommets la face visible courante dans le rep camera
                sommet0 = arr_coords_cam[face[0],:]
                sommet1 = arr_coords_cam[face[1],:]
                sommet2 = arr_coords_cam[face[2],:]
                ## surface 3D
                surface3D_visible = surface3D_visible + calculer_aire_triangle_3D(sommet0, sommet1, sommet2)   

            ######################################################################################################   
            ## Sauvegarde des données
            obj_filename = os.path.join(directory_output,name+"_cam"+str(k)+"_v2.obj")
            write_obj_with_color(array_coords_camera, array_faces, vrai_idx_vert_visible, obj_filename)

            ## Sauvegarde des données
            arrays_output_path = os.path.join(directory_output,name+"_cam"+str(k)+"_metadata_arrays.npz")
            values_output_path = os.path.join(directory_output,name+"_cam"+str(k)+"_metadata_values.pkl")

            # Step 1: Save all array data in a compressed .npz file
            np.savez_compressed(
                arrays_output_path,
                centroid=centroid,
                array_coords_camera=array_coords_camera,
                array_normals_camera=array_normals_camera,
                array_normals_camera_norm=array_normals_camera_norm,
                array_normals_face_camera=array_normals_face_camera,
                array_pixels=array_pixels,
                dephtMap=dephtMap,
                idx_faces_visibles=idx_faces_visibles,
                idx_vert_visible=idx_vert_visible,
                vrai_idx_vert_visible=vrai_idx_vert_visible,
                vrai_idx_faces_visibles=vrai_idx_faces_visibles,
                surface3D=surface3D,
                surface3D_visible=surface3D_visible,
                arr_cos_vert=arr_cos_vert)
            #print(values_output_path, "npz ok")

            # Step 2: Save scalar values and metadata in a separate file using pickle
            metadata = {
                "camera_k": k, 
                "nb_vertices": nb_vertices, "nb_faces": nb_faces,
                "diagonal": diagonal,
                "data_blender_cam": data_blender_cam,
                "focale": focale,
                "W_image": W_image, "H_image": H_image,
                "cam": cam, "lookat": lookat, "up": up,
                "WorldToCamera": WorldToCamera}
            
            with open(values_output_path, "wb") as f: pickle.dump(metadata, f)

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
results = process_files_in_parallel(mesh_paths, max_workers=14)

# Record the end time
end_time = time.time()
# Calculate and print the running time
running_time = end_time - start_time
print(f"Running time: {running_time:.2f} seconds for {len(mesh_paths)} files")

# Write results to file
with open("/home/pelissier/These-ATER/Papier_international3/Dataset/error_run1_projection_remeshing1.txt", "w") as file:
    for name, path in results:
        file.write(f"{name}: {path}\n")
            
            

