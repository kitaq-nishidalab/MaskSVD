import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import open3d as o3d

class PointCloudProcessor:
    def __init__(self):
        rospy.init_node('point_cloud_processor', anonymous=True)
        self.sub = rospy.Subscriber('/camera/depth/color/points', PointCloud2, self.point_cloud_callback, queue_size=1)
    
    def point_cloud_callback(self, msg):
        # Convert ROS PointCloud2 to a numpy array
        cloud_points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        if not cloud_points:
            rospy.logwarn("No points in the received PointCloud2 message.")
            return
        
        # Convert to Open3D PointCloud
        point_cloud_np = np.array(cloud_points)
        o3d_cloud = o3d.geometry.PointCloud()
        o3d_cloud.points = o3d.utility.Vector3dVector(point_cloud_np[:, :3])

        # Estimate normals for the point cloud
        try:
            o3d_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
            o3d_cloud.orient_normals_consistent_tangent_plane(100)
            
            # Perform Poisson reconstruction
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(o3d_cloud, depth=8)
            
            # Compute normals for the mesh (required for STL export)
            mesh.compute_vertex_normals()

            # Save the mesh as a binary STL file
            o3d.io.write_triangle_mesh("table.stl", mesh, write_ascii=False)
            rospy.loginfo("Mesh saved as table.stl")
        
        except Exception as e:
            rospy.logerr(f"Error during meshing: {e}")

def main():
    processor = PointCloudProcessor()
    rospy.spin()

if __name__ == '__main__':
    main()
