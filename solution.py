import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from sklearn.cluster import KMeans,DBSCAN
from sklearn.metrics import silhouette_score


pcd = o3d.io.read_point_cloud("FARO_Scan_Full_Sub1_down3_with_normals.ply")

puntos = np.asarray(pcd.points)
colores = np.asarray(pcd.colors)
normales = np.asarray(pcd.normals)

#o3d.visualization.draw_geometries([pcd])

print(f"puntos:{puntos} \n") # [[x1,y1,z1],[x2,y2,z2]...[xn,yn,zn]]
print(f"colores:{colores} \n") # [[r1,g1,b1],[r2,g2,b2]...[rn,gn,bn]]
print(f"normales:{normales} \n") # [[nx1,ny1,nz1],[nx2,ny2,nz2]...[nxn,nyn,nzn]]

N = len(puntos) ## numero de puntos
clase = np.array(["falta clasificacion"] * N,dtype=object) ## array donde a cada punto se le asigna su clasificación (string)
indices = np.full(N,-1,dtype=int) ## array donde guarda el numero del objeto que pertenece cada punto
next_id = 0 ## siguiente codigo de un cluster
floor_z = -1 ## altura del piso

## ---------------------------------------------------------------------------------------------- ##

## Paso 1: superficies horizontales
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
## Supera el umbral de centros[k_z]>= 0.8 . Si lo es se marca como true, si no solamente como false
horizontal = (kmeans.labels_ == k_z) if centros[k_z] >= 0.8 else np.zeros(N, bool)
## En horizontal se tiene los puntos en superficiones horizontales


## Paso 2: piso y techos
print("Paso 2: k means sobre superficies horizontales para determinar el piso y techo")
## Puntos que se asignaron a piso o a techo
## Para que los siguientes pasos ya no los usen
usado = np.zeros(N, bool)

## idx_h tiene indices de los puntos en superficies horizontales
idx_h = np.where(horizontal)[0]

## z = altura de cada punto en la superficie horizontal 
z = puntos[idx_h, 2].reshape(-1, 1)

## Para realizar el kmeans, vamos a iterar por varios k. Con la metrica de silloutte
## Se va ha elegir el mejor. Por default es k = 2 y silloutte = -1. 
## Y se va ha iterar hasta k = 6.
mejor_K, mejor_s = 2, -1

for K in range(2,  7):
    prueba = KMeans(n_clusters= K, n_init=10, random_state=0).fit_predict(z)
    silhouette_i = silhouette_score(z, prueba,  sample_size=min(20000, len(z)), random_state=0)
    print(f"  K={K}  silueta={silhouette_i:.3f}")
    if silhouette_i > mejor_s: mejor_K, mejor_s = K, silhouette_i

## Una vez elegido el k adecuado volvemos hacer k means-final
km_z = KMeans(n_clusters=mejor_K, n_init=10, random_state=0).fit(z)
## La variable niveles guarda los centros de los grupos, osea la altura "representativa" de cada grupo
niveles = km_z.cluster_centers_.ravel()

## Para el piso debe ser la altura minima, el nivel más bajo
## Lo mismo para techos, niveles más altos
k_piso, k_techo = int(np.argmin(niveles)) , int(np.argmax(niveles))

## Se va iterar 2 veces una para piso y la otra para el techo
for nombre, k in [("piso",k_piso),("techo",k_techo)]:
    ## Puntos en superficies horizontales cuyo grupo es k 
    ## idx_h los convierte en posiciones de la nube
    puntos_miembros = idx_h[km_z.labels_ == k]

    ## Altura tipica de un nivel (piso o techo) mediana
    z_med = np.median(puntos[puntos_miembros,2])

    ## Se selecciona solo los puntos a +-0.15 de esa altura tipica de un nivel
    ## Para no ser tan restrictivos
    sel = puntos_miembros[np.abs(puntos[puntos_miembros,2] - z_med) <= 0.15]

    ## Se guarda el resultado
    clase[sel] = nombre # se asigna el nombre del grupo a los puntos seleccionados
    indices[sel] = next_id # numero de objeto donde piso = 0 y techo va ha ser 1
    usado[sel] = True # los puntos seleccionados ya no deben ser procesados

    if nombre == "piso":
        floor_z = z_med  # altura del piso
    next_id += 1 


## Paso 3: dbscan para otros clusters
print("Paso 3, se va ha usar dbscan para identificar ruido sobre xyz")

## Se seleccionan los puntos que aun no tienen cluster
resto = np.where(~usado)[0]

## d_nn contiene para cada punto la distancia a su vecino mas cercano
## osea la separacion entre puntos de la nube 
d_nn = np.asarray(pcd.compute_nearest_neighbor_distance())

## con eso se elige un epsilon, que sera 3 veces la separacion tipica
## 3 veces por que con 3 nos aseguramos que unir puntos aunque hayan pequeños huecos
## pero sin saltar hacia otro cluster
eps = 3 * float(np.median(d_nn))
print(f"mediana dist. vecino = {np.median(d_nn):.4f} m -> eps = {eps:.4f} m, minPts = {10}")

## con el epsilon elegido se hace el DBSCAN, se elige minPts como 10 pues ese valor deja
## que una superficie "delgada" (pared o algo asi) forme un cluster.
## ojo que solo se pasan puntos que no aun no tienen cluster, .labels_ le da un cluster a cada punto
## o tmb puede decir que el punto sea ruido
lab = DBSCAN(eps=eps, min_samples=10, n_jobs=-1).fit(puntos[resto]).labels_

## Se imprime cuantos clusters quedaron: lab.max() + 1. 
## Y cuantos de los puntos son ruido para el DBSCAN
print("clusters:", lab.max() + 1, "| ruido:", (lab == -1).sum())