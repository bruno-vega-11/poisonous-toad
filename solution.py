import open3d as o3d
import numbers as np

pcd = o3d.io.read_point_cloud("FARO_Scan_Full_Sub1_down3_with_normals.ply")
o3d.visualization.draw_geometries([pcd])