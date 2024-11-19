import math
import numpy as np
import sys
from matplotlib import cm
import os
import pandas as pd

from utils import cross, dotproduct
  
def write_obj_with_color(vertices, faces, indices, obj_filename):
    # Open the OBJ file for writing
    with open(obj_filename, "w") as obj_file:     
        # Write each vertex to the OBJ file
        for i, v in enumerate(vertices):
            # Use the red material for vertices in the red_indices list
            if i in indices:
                obj_file.write(f"v {v[0]} {v[1]} {v[2]} 255 0 0\n")
            else:
                obj_file.write(f"v {v[0]} {v[1]} {v[2]} 0 0 0\n")
    
        for face in faces:
            # OBJ format uses 1-based indexing, so we add 1 to the vertex indices
            obj_file.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")
            
####################################################
####################################################
## PAPIER 2
####################################################
# Positon 3D de la caméra + coord 3D du POI + vecteur 3D up
# cam : position caméra dans repère monde 
# Retour la matrice de taille 4x4 
def get_worldTocamera_matrix(cam, lookat, up):
    ## Calculs des vecteurs du repère Caméra
    # vecteur de direction 
    d = lookat - cam
    ## repère caméra
    z_c = d/np.linalg.norm(d)
    x_c = cross(up, d)
    x_c = x_c/np.linalg.norm(x_c)
    y_c = cross(z_c, x_c)

    ## Matrice de passage
    WorldToCamera = np.zeros((4,4))
    WorldToCamera[0,0:3] = x_c
    WorldToCamera[1,0:3] = y_c
    WorldToCamera[2,0:3] = z_c

    ## vecteur translation 
    R = WorldToCamera[:3,:3]
    T = np.dot(R, cam)

    WorldToCamera[:3,3] = -1*T
    WorldToCamera[3,3] = 1
    return WorldToCamera

# a_coords de taille nb_sommetx3
# matrix : matrice qu permet de passer du repère monde au repère caméra
def monde_to_camera(matrix, a_coords):
    nb_sommet = a_coords.shape[0]
    ## Projection des sommets dans le repère caméra
    array_coords_RepCamera = np.ones((nb_sommet,4))
    for i in range(nb_sommet):
        vertice_i = a_coords[i,:]
        vert = np.ones(4)
        vert[:3] = vertice_i
        projected_point = np.dot(matrix, vert)  
        array_coords_RepCamera[i,:] = projected_point
    return array_coords_RepCamera

# Prend en entrée : la position de la caméra (coord 3D [x,y,z]), POI (coord 3D [x,y,z]), vecteur up
# Positon 3D de la caméra + coord 3D du POI + vecteur 3D up
# a_coords de taille nb_sommetx3
def world_to_camera(cam, lookat, up, a_coords):
    #matrice qu permet de passer du repère monde au repère caméra
    matrix = get_worldTocamera_matrix(cam, lookat, up)
    return monde_to_camera(matrix, a_coords)

# Matrice Barycentre d'une face avec comme sommet 2D : f0, f1 et f2
def BarycentreMatrice(f0,f1,f2):
    C = np.zeros((3,3))
    x1 = f0[0]
    y1 = f0[1]
    x2 = f1[0]
    y2 = f1[1]
    x3 = f2[0]
    y3 = f2[1]

    C[0][0] = x2*y3-x3*y2
    C[0][1] = y2-y3
    C[0][2] = x3-x2

    C[1][0] = x3*y1-x1*y3
    C[1][1] = y3-y1
    C[1][2] = x1-x3

    C[2][0] = x1*y2-x2*y1
    C[2][1] = y1-y2
    C[2][2] = x2-x1

    A = x2*y3 - x3*y2 + x3*y1 - x1*y3 + x1*y2 - x2*y1
    return C/A

def BoundingBox(f0,f1,f2,width,height):
    xmin = int(min(f0[0],f1[0],f2[0]))
    xmax = int(max(f0[0],f1[0],f2[0]))
    ymin = int(min(f0[1],f1[1],f2[1]))
    ymax = int(max(f0[1],f1[1],f2[1]))

    # Si on sort de l'image # A mettre jolie dans une fonction à part
    xmin = max(0,xmin)
    xmax = min(width-1,xmax)
    ymin = max(0,ymin)
    ymax = min(height-1, ymax)

    return xmin,ymin,xmax,ymax

# A partir de coord 3D des sommets dans les repère caméra, on projete en 2D
# a_coords de taille nb_sommetx3
# retourne une matrice nb_sommet*3 avec : les coord 2D des sommets et la VRAIE profondeur
def camera_to_image(focale, W_image, H_image, a_camera):
    ## Matrice de projection et de calibrage
    projection = np.zeros((3,4))
    projection[0,0] = 1; projection[1,1] = 1; projection[2,2] = 1
    calibration = np.eye(3)
    calibration[0,0] = focale; calibration[1,1] = focale; calibration[0,2] = W_image/2; calibration[1,2] = H_image/2

    ## Coorodnnées 2D des sommets + VRAIE profondeur
    nb_sommet = a_camera.shape[0]
    Pixels = np.zeros((nb_sommet,3))
    for i in range(nb_sommet):
        pixel =  np.dot(calibration, np.dot(projection, a_camera[i,:]))
        Pixels[i,:2] = pixel[:2]/pixel[2]
        Pixels[i,2] = pixel[2] # vraie profondeur
    return Pixels


def get_visible_vertices(W_image, H_image, a_faces, a_pixels, a_camera, a_normal, que_depthmap=False):
    # Initialisation carte de profondeur
    dephtMap = np.array([[ float('inf') for i in range(W_image)] for j in range(H_image)])
    indices = np.array([[ -1 for i in range(W_image)] for j in range(H_image)])
    # Pour chaque face
    for j in range(a_faces.shape[0]):
        currFace = a_faces[j,:] # vecteur de 3 entiers
        # les INDICES des 3 sommets de currFace. 3 entiers
        v0 = int(currFace[0]); v1 = int(currFace[1]); v2 = int(currFace[2])
        # COORD 2D de ces 3 sommets, dans plan image. Vecteur taille (2,1)
        f0 = a_pixels[v0,:2];  f1 = a_pixels[v1,:2]; f2 = a_pixels[v2,:2]
        # float --> Entier
        f0[0] =  round(f0[0]); f0[1] =  round(f0[1])
        f1[0] =  round(f1[0]); f1[1] =  round(f1[1])
        f2[0] =  round(f2[0]); f2[1] =  round(f2[1])
        # COORD 3D des 3 sommets, dans repère camera. Vecteur taille (3,1)
        p0 = a_camera[v0,:3]; p1 = a_camera[v1,:3]; p2 = a_camera[v2,:3]
        # PROFONDEUR des 3 sommets. 3 Float
        dv0 = p0[2]; dv1 = p1[2]; dv2 = p2[2]
        # Matrice du barycntre
        C = BarycentreMatrice(f0,f1,f2)
        # Profondeur minimale des sommets
        minDFace = min(dv0,dv1,dv2)
        # Bounding Box
        xmin,ymin,xmax,ymax = BoundingBox(f0,f1,f2,W_image,H_image)
        for xx in range(xmin,xmax+1):
            for yy in range(ymin,ymax+1):
                # Si la profondeur est plus grande, on passe au pixel suivant
                if minDFace > dephtMap[yy,xx]:  continue
                # Coordnnées barycentrique du pixel courant
                coord_bary = np.dot(C,np.array([1, xx, yy]))
                # On arrondie
                alpha = round(coord_bary[0],6)
                beta = round(coord_bary[1],6)
                gamma = round(coord_bary[2],6)
                # Si le point est hors du triangle 
                if not ((alpha >= 0) & (beta >= 0) & (gamma >= 0)): continue
                # Profondeur
                dd = alpha * dv0 + beta*dv1 + gamma*dv2
                if ((dd <= 0) or (dd > dephtMap[yy, xx])): continue
                # Remplissage de la carte de profondeur
                dephtMap[yy, xx] = dd
                # Si le pixel que l'on rajoute est un des 3 vrais sommets de la face courante, 
                # on met son indice dans les tableau 
                if not(que_depthmap) :
                    if ((xx == f0[0]) and (yy == f0[1])): 
                        rayon = -1*a_camera[v0,:]
                        rayon = rayon/np.linalg.norm(rayon)
                        normal = a_normal[v0,:]
                        # sommet pas au bord du visible
                        if dotproduct(rayon, normal) >= 0:
                            indices[xx, yy] = v0

                    elif ((xx == f1[0]) and (yy ==f1[1])): 
                        rayon = -1*a_camera[v1,:]
                        rayon = rayon/np.linalg.norm(rayon)
                        normal = a_normal[v1,:]
                        # sommet pas au bord du visible
                        if dotproduct(rayon, normal) >= 0:
                            indices[xx, yy] = v1

                    elif ((xx == f2[0]) and (yy ==f2[1])):
                        rayon = -1*a_camera[v2,:]
                        rayon = rayon/np.linalg.norm(rayon)
                        normal = a_normal[v2,:]
                        # sommet pas au bord du visible
                        if dotproduct(rayon, normal) >= 0: 
                            indices[xx, yy] = v2

                    else : indices[xx, yy] = -1

    return dephtMap, list(indices[np.where(indices!=-1)]), indices

## Filtrage avec interpolation
def filtrage_faces_occultees2(array_centre_faces, indices_faces_visibles, carte_profondeur, foc, W_img, H_img, epsilon_z):
    vraie_indices_faces_visibles = []
    #print('filtrage version2')
    for idx_f in indices_faces_visibles:
        # centre face visible
        centre_f = array_centre_faces[idx_f]
        # projection centre en 2D
        centre_f_augmented = np.ones(4)
        centre_f_augmented[:3] = centre_f
        centre_2D = camera_to_image(foc, W_img, H_img, np.array([centre_f_augmented]))[0]
        ## Coord IJ qui entoure le centre 2D
        i_avant = int(centre_2D[1]); i_apres = i_avant+1; alpha_i = centre_2D[1] - i_avant
        j_avant = int(centre_2D[0]); j_apres = j_avant+1; alpha_j = centre_2D[0] - j_avant
        ## Première interpolation entre (i,j) et (i+1,j) 
        z_avant = alpha_i*carte_profondeur[i_avant, j_avant] + (1-alpha_i)*carte_profondeur[i_apres,j_avant]
        ## Deuxième interpolation entre (i,j+1) et (i+1,j+1)     
        z_apres = alpha_i*carte_profondeur[i_avant, j_apres] + (1-alpha_i)*carte_profondeur[i_apres,j_apres]
        ## Troisième interpolation 
        z_depthmap = alpha_j*z_avant + (1-alpha_j)*z_apres
        ## Est ce que le centre projeté est bien celui sur la carte de profondeur
        if abs(z_depthmap - centre_2D[2]) <= epsilon_z : vraie_indices_faces_visibles.append(idx_f)
    return vraie_indices_faces_visibles


def find_adjacent_faces(faces):
    adjacent_faces = {}

    # Parcourir chaque face
    for face_index, face in enumerate(faces):
        # Pour chaque combinaison de sommets de la face (i, j, k)
        for i in range(3):
            j = (i + 1) % 3
            k = (i + 2) % 3

            # Créer une clé unique pour la paire de sommets (i, j)
            edge_key = (face[i], face[j])

            # Si la clé n'est pas déjà dans le dictionnaire, l'initialiser avec une liste vide
            if edge_key not in adjacent_faces:
                adjacent_faces[edge_key] = []

            # Ajouter l'indice de la face actuelle à la liste des faces adjacentes
            adjacent_faces[edge_key].append(face_index)

    return adjacent_faces

def find_faces_for_vertices(adjacent_faces):
    faces_for_vertices = {}

    # Parcourir les arêtes (paires de sommets) du dictionnaire adjacent_faces
    for edge, face_indices in adjacent_faces.items():
        vertex1, vertex2 = edge

        # Ajouter les indices des faces aux sommets correspondants
        if vertex1 not in faces_for_vertices:
            faces_for_vertices[vertex1] = set()
        faces_for_vertices[vertex1].update(face_indices)

        if vertex2 not in faces_for_vertices:
            faces_for_vertices[vertex2] = set()
        faces_for_vertices[vertex2].update(face_indices)

    return faces_for_vertices

def find_vertices_with_visible_faces(faces_for_vertices, visible_faces):
    vertices_with_visible_faces = set()

    # Parcourir les sommets et les indices des faces associées
    for vertex_index, face_indices in faces_for_vertices.items():
        # Vérifier si au moins une face adjacente est visible
        for face_index in face_indices:
            if face_index in visible_faces:
                # Ajouter le sommet à l'ensemble des sommets avec des faces visibles
                vertices_with_visible_faces.add(vertex_index)
                break  # Sortir de la boucle intérieure si une face est visible

    return vertices_with_visible_faces

def compute_triangle_center(vertex1, vertex2, vertex3):
    # Calculate the average of x, y, and z coordinates of the vertices
    center_x = (vertex1[0] + vertex2[0] + vertex3[0]) / 3
    center_y = (vertex1[1] + vertex2[1] + vertex3[1]) / 3
    center_z = (vertex1[2] + vertex2[2] + vertex3[2]) / 3

    # Return the coordinates of the center as a tuple
    return [center_x, center_y, center_z]

def compute_face_centre(face, arr_vertices):
    sommet1 = arr_vertices[face[0],:]
    sommet2 = arr_vertices[face[1],:]
    sommet3 = arr_vertices[face[2],:]
    return compute_triangle_center(sommet1, sommet2, sommet3)   

def idx_faces_from_idx_vert(arr_f, indice_vert):
    return([idx_f for idx_f in range(arr_f.shape[0]) if (indice_vert in arr_f[idx_f,:])])


def calculer_aire_triangle_3D(coord_sommet1, coord_sommet2, coord_sommet3):
    # Calcul des vecteurs entre les sommets
    vecteur1 = [coord_sommet2[0] - coord_sommet1[0], coord_sommet2[1] - coord_sommet1[1], coord_sommet2[2] - coord_sommet1[2]]
    vecteur2 = [coord_sommet3[0] - coord_sommet1[0], coord_sommet3[1] - coord_sommet1[1], coord_sommet3[2] - coord_sommet1[2]]

    # Calcul du produit vectoriel des deux vecteurs
    produit_vectoriel = [vecteur1[1] * vecteur2[2] - vecteur1[2] * vecteur2[1],
                         vecteur1[2] * vecteur2[0] - vecteur1[0] * vecteur2[2],
                         vecteur1[0] * vecteur2[1] - vecteur1[1] * vecteur2[0]]

    # Calcul de la norme du produit vectoriel
    norme_produit_vectoriel = math.sqrt(produit_vectoriel[0] ** 2 + produit_vectoriel[1] ** 2 + produit_vectoriel[2] ** 2)

    # Calcul de l'aire du triangle
    aire_triangle = 0.5 * norme_produit_vectoriel

    return aire_triangle


