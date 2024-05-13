import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud("./pointcloud/Pointcloud_NEURAL.ply")
print(pcd)
# # o3d.visualization.draw_geometries([pcd])
# print(pcd.height)

# colors = pcd.colors
# print(np.asarray(colors))

out = o3d.io.write_point_cloud("./test.pcd", pcd, write_ascii=True)