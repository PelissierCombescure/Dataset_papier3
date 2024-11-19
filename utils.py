import numpy as np
import os

def read_paths_from_txt(txt_file):
    """Reads a list of file paths from a .txt file."""
    with open(txt_file, 'r') as f:
        paths = f.readlines()
    # Strip any whitespace characters like `\n` at the end of each line
    paths = [path.strip() for path in paths]
    return paths


# Recréer la strcuture de dossiers de ModelNet40 dans ModelNet40_centered_scaled
def replicate_structure(src_dir, dest_dir):
    # Walk through the directory tree of the source directory
    for root, dirs, _ in os.walk(src_dir):
        # Construct the destination directory path
        relative_path = os.path.relpath(root, src_dir)
        dest_path = os.path.join(dest_dir, relative_path)
        
        # Create the directories in the destination
        os.makedirs(dest_path, exist_ok=True)
        #print(f"Created: {dest_path}")
        
def get_info_path(path_modelnet40):
    categorie = path_modelnet40.split('/')[-3]
    type = path_modelnet40.split('/')[-2]
    return categorie, type
        

########################################### OBJ

def read_obj_file(file_path):
    """Reads an OBJ file and returns vertices and faces."""
    vertices = []
    faces = []
    
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            # Read vertices
            if parts[0] == 'v':
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            # Read faces
            elif parts[0] == 'f':
                face = []
                for part in parts[1:]:
                    vertex_index = int(part.split('/')[0]) - 1  # Convert from 1-based to 0-based index
                    face.append(vertex_index)
                faces.append(face)
    
    return np.array(vertices), np.array(faces)

def write_obj_file(output_file_path, vertices, faces):
    """Writes the vertices and faces to a new OBJ file."""   
    with open(output_file_path, 'w') as f:
        for vertex in vertices:
            f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        for face in faces:
            # Add 1 to each index for OBJ format (1-based indexing)
            face_str = ' '.join(str(index + 1) for index in face)
            f.write(f"f {face_str}\n")
            
def get_centroid(a_faces, a_coords):
    area_total = 0
    centroid = np.array([0,0,0])
    for face in a_faces:
        v0 = a_coords[face[0],:]
        v1 = a_coords[face[1],:]
        v2 = a_coords[face[2],:]
        center = (v0+v1+v2)/3
        T_area = np.linalg.norm(cross(v1-v2, v0-v2))*0.5
        area_total = area_total + T_area
        centroid = centroid + T_area*center

    centroid = centroid/area_total
    return centroid

def compute_bounding_box_diagonal(corners):
    """
    Compute the diagonal of a 3D bounding box given its 8 corner coordinates.
    
    Parameters:
    corners (numpy array): An array of shape (8, 3) containing the 3D coordinates of the 8 corners.
    
    Returns:
    float: The length of the diagonal of the bounding box.
    """
    # Calculate the minimum and maximum coordinates across all points
    min_corner = np.min(corners, axis=0)
    max_corner = np.max(corners, axis=0)
    
    # Compute the Euclidean distance between the minimum and maximum corners
    diagonal_length = np.linalg.norm(max_corner - min_corner)
    
    return diagonal_length

############################################# MATH

def cross(a, b):
    c = np.array([a[1]*b[2] - a[2]*b[1],
         a[2]*b[0] - a[0]*b[2],
         a[0]*b[1] - a[1]*b[0]])

    return c

def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))


def normalisation_vecteur(vecteur):
    return vecteur/np.linalg.norm(vecteur)

def normalisation(L_vecteur):
    return list(map(normalisation_vecteur, L_vecteur))