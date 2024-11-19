# Files

--- 

`apply_transform.ipynb` : center and scale a obj file. Can do it for a list of obj file.

---

`create_dodecahedral`

---

`get_files.ipynb` : A partir du chemin d'un dossier et d'une extension (ex : obj, off...), écrit un fichier contenant tous les chemins absolus des fichiers d'extension demandée contenu dans le dossier. La recherche est **récursive**.

<span style="color:pink"> *outputs* : </span> txt file as : obj_files_ModeleNet40.txt or off_files_ModeleNet40.txt 

<span style="color:pink"> *inputs* </span>: folder path (str) & extension (str)

<span style="color:pink"> *Conda env* </span>: ok avec `PoinNet0_env`

---

`projections.ipynb`: notebook to generate {depthmap, visible vertices} from blender camera coordinates. Funcions used to World --> Camera --> Pixels are in `fonction_PtofView.py`
Version  script pyhton `run_projections.py`

<span style="color:pink"> *outputs :*</span> Folder ->  **Rendu/ModelNet40/my_circular_12_elevation_30_remeshing**

For each 12 views/objet (3 x 12 x 12311 files)

-  npz file : *NAME+"_cam"+NUM+"_metadata_arrays.npz*

```
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
```

- pkl file : *NAME+"_cam"+NUM+"_metadata_values.pkl*
  
  ```
  metadata = {
          "camera_k": k, 
          "nb_vertices": nb_vertices, "nb_faces": nb_faces,
          "diagonal": diagonal,
          "data_blender_cam": data_blender_cam,
          "focale": focale,
          "W_image": W_image, "H_image": H_image,
          "cam": cam, "lookat": lookat, "up": up,
          "WorldToCamera": WorldToCamera}
  ```

- OBJ file

---

---

# paths_files/

--- 

`{obj,off}_{∅, SMPL, SIMPLER}_files_ModelNet40{∅, _centered_scaled}.txt` : txt file with the 12311 paths of meshes of *ModelNet40* or *ModelNet40_centered_scaled* according to extension (*obj or off*) and version (*None, SMPL or SMPLER*)

--- 

---

# Rendu/ModelNet40/

---
