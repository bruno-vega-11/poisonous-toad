import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from sklearn.cluster import KMeans

 

pcd = o3d.io.read_point_cloud("FARO_Scan_Full_Sub1_down3_with_normals.ply")

puntos = np.asarray(pcd.points)
colores = np.asarray(pcd.colors)
normales = np.asarray(pcd.normals)

#o3d.visualization.draw_geometries([pcd])

print(f"puntos:{puntos} \n") # [[x1,y1,z1],[x2,y2,z2]...[xn,yn,zn]]
print(f"colores:{colores} \n") # [[r1,g1,b1],[r2,g2,b2]...[rn,gn,bn]]
print(f"normales:{normales} \n") # [[nx1,ny1,nz1],[nx2,ny2,nz2]...[nxn,nyn,nzn]]

N = len(puntos) ## numero de puntos

## Paso 1: secciones horizontales
print("Paso 1: k-means sobre |nz|")
##    |nzi| cuanto apunta la normal del punto i hacia el eje z, valor abs = arriba o abajo
## Si |nzi| = 1 -> el punto esta en una superficie horizontal, la normal apunta arriba o abajo 
## Si |nzi| = 0 -> el punto esta en una superficie vertical, la normal apunta a los costados 
abs_nzi = np.abs(normales[:,2]).reshape(-1,1) # array donde cada fila = nzi
print("Componente z de las normales: ")
print(abs_nzi)
## Se va ha aplicar kmeans agrupando a sus puntos segun componente nz de la normal
## Son 3 grupos con la idea de más o menos segmentar por horizontales, verticales, intermedio
kmeans = KMeans(n_clusters = 3, n_init=10, random_state=0).fit(abs_nzi)
## Centroides son "representantes" de los clusters
centros = kmeans.cluster_centers_.ravel()
print("centros de |nz| (ordenados):", np.sort(centros).round(3))
## Nos intereza al grupo donde los puntos esten en superficies horizontales
## Osea el grupo cuyo centro sea aprox 1
## Sacamos ese numero de grupo
k_z = int(np.argmax(centros))
## Igual se chequea que ese clusters sea realmente horizontal
## Por ejemplo solo se agarra se toma los puntos que pertenece al cluster k_z y además
## Supera el umbral de k_z >= 0.8 . Si lo es se marca como true, si no solamente como false
horizontal = (kmeans.labels_ == k_z) if centros[k_z] >= 0.8 else np.zeros(N, bool)
## En horizontal se tiene los puntos en superficiones horizontales


