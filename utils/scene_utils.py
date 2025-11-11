import torch
import os
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = ['Times New Roman']

import numpy as np
import copy
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import cv2

def get_depth_sudden_change_mask(depth,
                                 grad_trd = 1.0,
                                 sobel_k_size = 3,
                                 do_erode = False,
                                 do_dialte = False,
                                 do_blur = False,
                                # debug_vis_depth_sobel = False,
                                debug_vis_depth_sobel = True,
                                erode_kernel = np.ones((2, 2), np.uint8),
                                dilate_kernel = np.ones((3, 3), np.uint8)# better not scale the grads, intuitvely more reasonable

                                    ):
    assert sobel_k_size in [1,3,5,7]
    depth_sobel = cv2.Sobel(depth, cv2.CV_64F, 1,1,ksize = sobel_k_size, scale = 1)
    # depth_sobel_uint8 = cv2.convertScaleAbs(depth_sobel,alpha=255/depth_sobel.max(),)
    depth_sobel_abs = np.abs(depth_sobel)# better not scale the grads, intuitvely more reasonable 
    if do_blur: 
        depth_sobel_abs = cv2.GaussianBlur(depth_sobel_abs, (3, 3), 0)

    #thresholding a binary mask
    depth_sudden_change_mask = depth_sobel_abs>grad_trd
    # # erode to remove noise
    depth_sudden_change_mask_eroded = depth_sudden_change_mask
    if do_erode:
        depth_sudden_change_mask_eroded = cv2.erode(depth_sudden_change_mask.astype(np.uint8)*255,
                                                        erode_kernel,
                                                        iterations=1)>0
    depth_sudden_change_mask_dilated = depth_sudden_change_mask_eroded
    if do_dialte:
        depth_sudden_change_mask_dilated = cv2.dilate(depth_sudden_change_mask_eroded.astype(np.uint8)*255, 
                                                    dilate_kernel, 
                                                    iterations=1)>0
    # plot the depth and depth_sobel
    if debug_vis_depth_sobel:
        import matplotlib.pyplot as plt
        # cv2.imshow(
        #             f'depth---grad---naive_binary---erode_binary---erode_then_diatal_binary',
        #         np.concatenate([cv2.convertScaleAbs(depth,alpha=255/depth.max()),
        #                         cv2.convertScaleAbs(depth_sobel_abs,alpha=255/depth.max()),
        #                         depth_sudden_change_mask.astype(np.uint8)*255,
        #                         depth_sudden_change_mask_eroded.astype(np.uint8)*255,
        #                         depth_sudden_change_mask_dilated.astype(np.uint8)*255,
        #                         ],axis=1),
        #         )
        
        #alpha blend the depth_sudden_change_mask_dilated with depth image with blueish color for the overlay binary mask
        depth_sudden_change_mask_dilated_overlay = np.zeros_like(depth).astype(np.uint8)
        depth_sudden_change_mask_dilated_overlay[depth_sudden_change_mask_dilated] = 255
        # depth_sudden_change_mask_dilated_overlay = cv2.merge([np.zeros_like(depth),np.zeros_like(depth),depth_sudden_change_mask_dilated_overlay])
        #make sure the two image are the same shape

        depth_overlay = cv2.addWeighted(cv2.convertScaleAbs(depth,alpha=255/depth.max()),0.5,depth_sudden_change_mask_dilated_overlay,0.5,0)
        cv2.imshow('depth_overlay',depth_overlay)
        # print(f'depth {depth.mean()} {depth.max()} {depth.min()}, grad max {depth_sobel_abs.max()} min {depth_sobel_abs.min()} binary trd {grad_trd}')
        cv2.waitKey(10)
    return depth_sudden_change_mask_dilated

            
            
def get_vertices_from_min_max_bounds(min_bounds,max_bounds,R = None,center = None, conduct_local2world = False):
    # if conduct_local2world:
        # assert R != None
        # assert center != None
    # Create box vertices in local coordinate system
    vertices_local = np.array([
        [min_bounds[0], min_bounds[1], min_bounds[2]],
        [max_bounds[0], min_bounds[1], min_bounds[2]],
        [max_bounds[0], max_bounds[1], min_bounds[2]],
        [min_bounds[0], max_bounds[1], min_bounds[2]],
        [min_bounds[0], min_bounds[1], max_bounds[2]],
        [max_bounds[0], min_bounds[1], max_bounds[2]],
        [max_bounds[0], max_bounds[1], max_bounds[2]],
        [min_bounds[0], max_bounds[1], max_bounds[2]]
    ])
    if conduct_local2world:
        # vertices = np.dot(R, vertices_local.T).T + center
        # Transform: rotate then translate
        vertices = (vertices_local @ R.T) + center
    return vertices

def visualize_6d_range_filtering_oriented_redundant(samples_xyz, points_inside_init_6d_range, min_xyz, max_xyz, track_id, current_iter, max_iter,
                                    ):

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import cv2
    from io import BytesIO
    
    # Create figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    inside_points = samples_xyz[points_inside_init_6d_range][:,0,:]
    if len(inside_points) > 0:
        ax.scatter(inside_points[:, 0], inside_points[:, 1], inside_points[:, 2], 
                c='green', alpha=0.6, label='Inside Range')
    
    outside_points = samples_xyz[~points_inside_init_6d_range][:,0,:]
    if len(outside_points) > 0:
        ax.scatter(outside_points[:, 0], outside_points[:, 1], outside_points[:, 2], 
                c='red', alpha=0.6, label='Outside Range')
    



    # Compute oriented bounding box
    print('Compute Once?')
    
    

    center, R, min_bounds, max_bounds, valid_mask, filtered_points = clustering_and_get_BBox_via_PCA(inside_points, 
                                                                cluster_eps=2.,
                                                                # cluster_eps='auto',
                                                                margin_scale=0.,
                                                                min_samples = 1,# HERE WE SET TO 1; BUT WHEN INIT WE CAN SET TO #OBJS
                                                                )

     # Add filtered points from DBSCAN
    if filtered_points is not None:
        ax.scatter(filtered_points[:, 0], filtered_points[:, 1], filtered_points[:, 2],
                  c='yellow', alpha=0.8, s=30, marker='*', label='DBSCAN Core Points')
    
    
    # Create box vertices in local coordinate system
    vertices_local = np.array([
        [min_bounds[0], min_bounds[1], min_bounds[2]],
        [max_bounds[0], min_bounds[1], min_bounds[2]],
        [max_bounds[0], max_bounds[1], min_bounds[2]],
        [min_bounds[0], max_bounds[1], min_bounds[2]],
        [min_bounds[0], min_bounds[1], max_bounds[2]],
        [max_bounds[0], min_bounds[1], max_bounds[2]],
        [max_bounds[0], max_bounds[1], max_bounds[2]],
        [min_bounds[0], max_bounds[1], max_bounds[2]]
    ])

    # vertices = np.dot(R, vertices_local.T).T + center
    # Transform: rotate then translate
    vertices = (vertices_local @ R.T) + center


    # Plot oriented bounding box edges

    edges = [
        (0,1), (1,2), (2,3), (3,0),  # Bottom face
        (4,5), (5,6), (6,7), (7,4),  # Top face
        (0,4), (1,5), (2,6), (3,7)   # Connecting edges
    ]
    for start, end in edges:
        ax.plot3D(*zip(vertices[start], vertices[end]), 
                color='blue', alpha=0.5, linewidth=2)
    
    # Plot coordinate axes at box center
    axis_length = np.max(max_bounds - min_bounds) * 0.5
    for i in range(3):
        direction = R[i] * axis_length
        ax.quiver(center[0], center[1], center[2],
                direction[0], direction[1], direction[2],
                color=['r', 'g', 'b'][i], alpha=0.8)
    
    # Plot original AABB in light gray for comparison
    min_x, min_y, min_z = min_xyz.values.cpu().numpy()
    max_x, max_y, max_z = max_xyz.values.cpu().numpy()
    orig_vertices = np.array([
        [min_x, min_y, min_z], [max_x, min_y, min_z],
        [max_x, max_y, min_z], [min_x, max_y, min_z],
        [min_x, min_y, max_z], [max_x, min_y, max_z],
        [max_x, max_y, max_z], [min_x, max_y, max_z]
    ])
    for start, end in edges:
        ax.plot3D(*zip(orig_vertices[start], orig_vertices[end]), 
                color='green', 
                alpha=0.7, linestyle='--')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Tool {track_id} 6D Range Filtering\nIter {current_iter}/{max_iter}\n'
                f'Inside: {points_inside_init_6d_range.sum()}, '
                f'Outside: {(~points_inside_init_6d_range).sum()}')
    
    # Add legend
    ax.legend()
    
    # Save plot to memory buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    
    # Convert to OpenCV image
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    
    # Create window name
    window_name = f'ORIENTED: Tool {track_id} 6D Range'
    # window_name = f'Tool {track_id} 6D Range - Iter {current_iter}'
    
    # Show image in window
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, img)
    
    # Save plot to file
    save_dir = 'vis_6d_range'
    # os.makedirs(save_dir, exist_ok=True)
    # cv2.imwrite(f'{save_dir}/tool{track_id}_iter{current_iter:04d}.png', img)
    
    # Wait for a short time (10ms) and check for 'q' key to quit
    key = cv2.waitKey(10)
    if key == ord('q'):
        cv2.destroyAllWindows()



def plot_6d_bbox_with_pts(vertices,points_inside,points,
tool_adc_mode = '',
tool_id = '',
current_iter = '',
max_iter = '',
vis_always_stay = False,
):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    points_cpu = points.cpu().numpy() if isinstance(points,torch.Tensor) else points
    mask = points_inside.cpu().numpy() if isinstance(points,torch.Tensor) else points_inside
    if points_cpu.ndim == 3:
        points_cpu = points_cpu.squeeze(1)
    assert points_cpu.ndim == 2
    # assert 0,points_cpu[mask].shape
    ax.scatter(
                # points_cpu[mask][:,0,0], 
                # points_cpu[mask][:,0,1],
                # points_cpu[mask][:,0,2], 

                points_cpu[mask][:,0], 
                points_cpu[mask][:,1],
                points_cpu[mask][:,2],             
            c='g', alpha=0.6, label='Inside')
    
    ax.scatter(
                # points_cpu[~mask][:,0,0],
                # points_cpu[~mask][:,0,1],
                # points_cpu[~mask][:,0,2],

                points_cpu[~mask][:,0],
                points_cpu[~mask][:,1],
                points_cpu[~mask][:,2], 
                c='r', alpha=0.6, label='Outside')
    
    # Plot box
    edges = [(0,1), (1,2), (2,3), (3,0),  # Bottom
            (4,5), (5,6), (6,7), (7,4),  # Top
            (0,4), (1,5), (2,6), (3,7)]  # Sides
    for start, end in edges:
        ax.plot3D(*zip(vertices[start], vertices[end]), color='b', alpha=0.5)
        
    ax.set_title(f' Tool{tool_id}: {tool_adc_mode}')
    # ax.set_title(f' Tool{tool_id}: {tool_adc_mode} outlier ratio {1-(float(points_inside.sum()/float(len(points))))} (Iter {current_iter}/{max_iter})')
    ax.legend()
    # plt.show(block=False)
    # # plt.pause(0.1)
    # plt.pause(10)


    import cv2
    from io import BytesIO
    
    # Save plot to memory buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    
    # Convert to OpenCV image
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    
    # Create window name
    window_name = f'Tool {tool_id} 6D Range'
    
    # Show image in window
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, img)

    # constantly keep the window
    if vis_always_stay:
        cv2.waitKey(0)
    else:
        # Wait for key press (10ms) and check for 'q' to quit
        key = cv2.waitKey(10)
        if key == ord('q'):
            cv2.destroyAllWindows()

def compute_pts_within_6D_box(samples_xyz, vertices,):
    """
    Check if points are inside an oriented bounding box using plane equations from vertices
    
    Args:
        samples_xyz: (N, M, 3) points to check
        vertices: (8, 3) vertices defining the oriented box
    Returns:
        points_inside: (N,) boolean mask of points inside box
    """
    device = samples_xyz.device
    points = samples_xyz[:,0,:]  # (N, 3)
    points_np = points.cpu().numpy()
    center = vertices.mean(axis=0)
    
    # Define the six faces of the box using vertices
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # front
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # back
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # bottom
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # top
        [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
        [vertices[1], vertices[2], vertices[6], vertices[5]]   # right
    ]
    
    # Check if points are inside all faces
    points_inside = np.ones(len(points), dtype=bool)
    for face in faces:
        v1, v2, v3 = face[:3]
        # Calculate normal
        normal = np.cross(v2 - v1, v3 - v1)
        normal = normal / (np.linalg.norm(normal) + 1e-6)
        
        # Make sure normal points outward
        face_center = np.mean(face, axis=0)
        if np.dot(normal, face_center - center) < 0:
            normal = -normal
            
        # Check which side of plane each point is on
        side = np.dot(points_np - v1, normal)
        points_inside &= (side <= 0)  # Inside if on negative side of all planes
    
                    
    points_inside = torch.from_numpy(points_inside).to(device)

    
    return points_inside    
 

def clustering_and_get_BBox_via_PCA(points, margin_scale=0.0, min_samples=1,
                                     cluster_eps = 0.5,
                                    #  cluster_only = False,
                                     
                                     ):
    
    def estimate_dbscan_eps(points, k=5, quantile=0.95):
        """
        Estimate DBSCAN eps parameter using k-nearest neighbor distances:
        1. Compute k-nearest neighbor distances for each point
        2. Use a quantile of these distances as eps
        """
        def plot_kdist_graph(points, k=5):
            """
            Plot k-distance graph using OpenCV window
            """
            import matplotlib.pyplot as plt
            import cv2
            from io import BytesIO
            
            # Create matplotlib figure
            plt.figure(figsize=(10, 5))
            
            # Compute k-distances
            nbrs = NearestNeighbors(n_neighbors=k).fit(points)
            distances, _ = nbrs.kneighbors(points)
            distances = np.sort(distances[:, 1:].mean(axis=1))
            
            # Plot
            plt.plot(distances)
            plt.ylabel(f'Mean {k}-neighbor distance')
            plt.xlabel('Points (sorted)')
            plt.title('K-distance Graph')
            plt.grid(True)
            
            # Save plot to memory buffer
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            plt.close()
            
            # Convert to OpenCV image
            buf.seek(0)
            img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            
            # Show image in OpenCV window
            window_name = 'K-distance Graph'
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, img)
            
            # Wait for key press (10ms) and check for 'q' to quit
            key = cv2.waitKey(10)
            if key == ord('q'):
                cv2.destroyAllWindows()


        # Fit nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k).fit(points)
        # Get distances to k-nearest neighbors for each point
        distances, _ = nbrs.kneighbors(points)
        # Sort distances in ascending order
        distances = np.sort(distances[:, 1:])  # Exclude distance to self
        # Use quantile of mean distances as eps
        eps = np.quantile(distances.mean(axis=1), quantile)
        
        return eps


    if torch.is_tensor(points):
        points = points.detach().cpu().numpy()
    
    try:
        if cluster_eps=='auto':
            # Estimate eps from point cloud
            cluster_eps = estimate_dbscan_eps(points, 
                                    k=5,  # Number of neighbors to consider
                                    quantile=0.95) 
            plot_kdist_graph(points, k=5)
            print('//////////////eps',cluster_eps)

        # DBSCAN clustering
        clustering = DBSCAN(eps=cluster_eps, min_samples=min_samples).fit(points)
        mask = clustering.labels_ != -1
        filtered_points = points[mask]


        
        # Center points
        center = np.mean(filtered_points, axis=0)
        centered_points = filtered_points - center
        
        # Compute PCA correctly
        covariance_matrix = np.cov(centered_points.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        
        # Sort eigenvalues in DESCENDING order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Make sure we have a right-handed coordinate system
        if np.linalg.det(eigenvectors) < 0:
            eigenvectors[:, 2] *= -1
            
        # Rotation matrix (eigenvectors as columns)
        R = eigenvectors
        
        # Project points to aligned frame
        points_in_obb_frame = (centered_points @ R)
        
        # Compute bounds
        min_bounds = np.min(points_in_obb_frame, axis=0)
        max_bounds = np.max(points_in_obb_frame, axis=0)
        
        if margin_scale > 0:
            extent = max_bounds - min_bounds
            margin = extent * margin_scale
            min_bounds -= margin
            max_bounds += margin
        # in local space
        return center, R, min_bounds, max_bounds, mask,filtered_points
        
    except Exception as e:
        print(f"OBB computation failed: {e}")
        return None, None, None, None, None,points               




@torch.no_grad()
def render_training_image(scene, gaussians, viewpoints, render_func, pipe, background, stage, iteration, time_now):
    def render(gaussians, viewpoint, path, scaling):
        # scaling_copy = gaussians._scaling
        render_pkg = render_func(viewpoint, gaussians, pipe, background, stage=stage)
        label1 = f"stage:{stage},iter:{iteration}"
        times =  time_now/60
        if times < 1:
            end = "min"
        else:
            end = "mins"
        label2 = "time:%.2f" % times + end
        image = render_pkg["render"]
        depth = render_pkg["depth"]
        image_np = image.permute(1, 2, 0).cpu().numpy()  # 转换通道顺序为 (H, W, 3)
        depth_np = depth.permute(1, 2, 0).cpu().numpy()
        depth_np /= depth_np.max()
        depth_np = np.repeat(depth_np, 3, axis=2)
        image_np = np.concatenate((image_np, depth_np), axis=1)
        image_with_labels = Image.fromarray((np.clip(image_np,0,1) * 255).astype('uint8'))  # 转换为8位图像
        # 创建PIL图像对象的副本以绘制标签
        draw1 = ImageDraw.Draw(image_with_labels)

        # 选择字体和字体大小
        font = ImageFont.truetype('./utils/TIMES.TTF', size=40)  # 请将路径替换为您选择的字体文件路径

        # 选择文本颜色
        text_color = (255, 0, 0)  # 白色

        # 选择标签的位置（左上角坐标）
        label1_position = (10, 10)
        label2_position = (image_with_labels.width - 100 - len(label2) * 10, 10)  # 右上角坐标

        # 在图像上添加标签
        draw1.text(label1_position, label1, fill=text_color, font=font)
        draw1.text(label2_position, label2, fill=text_color, font=font)
        
        image_with_labels.save(path)
    render_base_path = os.path.join(scene.model_path, f"{stage}_render")
    point_cloud_path = os.path.join(render_base_path,"pointclouds")
    image_path = os.path.join(render_base_path,"images")
    if not os.path.exists(os.path.join(scene.model_path, f"{stage}_render")):
        os.makedirs(render_base_path)
    if not os.path.exists(point_cloud_path):
        os.makedirs(point_cloud_path)
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    # image:3,800,800
    
    # point_save_path = os.path.join(point_cloud_path,f"{iteration}.jpg")
    for idx in range(len(viewpoints)):
        image_save_path = os.path.join(image_path,f"{iteration}_{idx}.jpg")
        render(gaussians,viewpoints[idx],image_save_path,scaling = 1)
    # render(gaussians,point_save_path,scaling = 0.1)
    # 保存带有标签的图像

    pc_mask = gaussians.get_opacity
    pc_mask = pc_mask > 0.1
    xyz = gaussians.get_xyz.detach()[pc_mask.squeeze()].cpu().permute(1,0).numpy()
    # visualize_and_save_point_cloud(xyz, viewpoint.R, viewpoint.T, point_save_path)
    # 如果需要，您可以将PIL图像转换回PyTorch张量
    # return image
    # image_with_labels_tensor = torch.tensor(image_with_labels, dtype=torch.float32).permute(2, 0, 1) / 255.0

def visualize_and_save_point_cloud(point_cloud, R, T, filename):
    # 创建3D散点图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    R = R.T
    # 应用旋转和平移变换
    T = -R.dot(T)
    transformed_point_cloud = np.dot(R, point_cloud) + T.reshape(-1, 1)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(transformed_point_cloud.T)  # 转置点云数据以匹配Open3D的格式
    # transformed_point_cloud[2,:] = -transformed_point_cloud[2,:]
    # 可视化点云
    ax.scatter(transformed_point_cloud[0], transformed_point_cloud[1], transformed_point_cloud[2], c='g', marker='o')
    ax.axis("off")
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')

    # 保存渲染结果为图片
    plt.savefig(filename)

def get_soft_mask_indices_prob(tool_mask, proj_2d, 
                               valid_indices,rows,cols,
                               num_gaussians,device, 
                                sigma=0.3,  # default keep rate for points outside mask
                                filter_mode=None,
                                current_iter=None,
                                max_iter=None,):
    def get_progressive_sigma(current_iter, max_iter, 
                            sigma_start=0.9,  # More permissive initially
                            sigma_end=0.1,
                            ):   # Stricter at the end
        """
        Calculate progressive keep rate for points outside mask
        Args:
            current_iter: Current iteration
            max_iter: Maximum iterations
            sigma_start: Starting keep rate (more permissive)
            sigma_end: Ending keep rate (stricter)
        Returns:
            Current keep rate for points outside mask
        """
        # Cosine annealing schedule
        progress = current_iter / max_iter
        keep_rate = sigma_end + 0.5 * (sigma_start - sigma_end) * (1 + np.cos(progress * np.pi))
        return keep_rate
    keep_rate_mode = 'trd'
    # keep_rate_mode = 'progress'
    dis_prob_sigma_mode = 'progress'
    dis_prob_sigma_mode = 'trd'
    # Get current keep rate
    # if progressive:
    if filter_mode == 'rand_prob':
        assert keep_rate_mode in ['trd','progress']
        keep_rate = get_progressive_sigma(current_iter, max_iter,
                                            sigma_start=0.9,  # More permissive initially
                            sigma_end=0.1,) if keep_rate_mode == 'progress' else sigma
    elif filter_mode == 'dis_prob':
        assert dis_prob_sigma_mode in ['trd','progress']
        dis_prob_sigma = 0.01  # Controls how quickly probability drops with distance, smaller, severer! better
        dis_prob_sigma = 0.2  # Controls how quickly probability drops with distance, smaller, severer! better
        # dis_prob_sigma = 0.5  # Controls how quickly probability drops with distance, smaller, severer! better
        # dis_prob_sigma = 5  # Controls how quickly probability drops with distance, smaller, severer! better
        dis_prob_sigma = get_progressive_sigma(current_iter, max_iter,
                                                sigma_start=0.8,  # More permissive initially
                            sigma_end=2.5,) if dis_prob_sigma_mode == 'progress' else dis_prob_sigma


    # Initialize all points as kept (True)
    # points_inside_2D_mask = torch.ones((num_gaussians), dtype=torch.bool).to(samples_xyz_in_cam.device)
    points_inside_2D_mask = torch.zeros((num_gaussians), dtype=torch.bool).to(device)
    
    # For valid points, check if they're outside mask and apply probability
    valid_points = proj_2d[valid_indices]
    valid_rows, valid_cols = valid_points[:, 1].cpu().numpy(), valid_points[:, 0].cpu().numpy()
    
    # Check which valid points are outside the mask
    tool_mask_np = tool_mask.cpu().numpy()
    points_outside_mask = ~tool_mask_np[valid_rows, valid_cols]
    
    if filter_mode == 'rand_prob':
        # Generate random values only for valid points that are outside mask
        random_values = np.random.random(len(valid_points))
        # Points outside mask have chance (1 - keep_rate) to be dropped
        points_to_keep = points_outside_mask & (random_values < keep_rate)

    elif filter_mode == 'dis_prob':
        # Convert mask to uint8 for distance transform
        mask_uint8 = tool_mask_np.astype(np.uint8) * 255
        # Compute distance transform
        dist_transform = cv2.distanceTransform(255 - mask_uint8, 
                                                cv2.DIST_L2, 5)
        # Normalize distances to [0, 1] range
        dist_transform = dist_transform / dist_transform.max()
        # Convert distances to probabilities (closer = higher probability)
        # Using exponential decay: prob = exp(-distance/dis_prob_sigma)
        prob_map = np.exp(-dist_transform / dis_prob_sigma)

        # prob_map = np.maximum(0, 1 - dist_transform/dis_prob_sigma)

        # Get probabilities for valid points
        point_probs = prob_map[valid_rows, valid_cols]

        # Generate random values and compare with distance-based probabilities
        random_values = np.random.random(len(valid_points))
        points_to_keep = points_outside_mask & (random_values < point_probs)
    else:
        assert 0, filter_mode



    
    # Extend the tool mask by adding the randomly kept points
    extended_mask = tool_mask_np.copy()
    extended_mask[valid_rows[points_to_keep], valid_cols[points_to_keep]] = True
    
    # Convert back to torch tensor
    tool_mask_soft = torch.from_numpy(extended_mask).to(tool_mask.device)

    points_inside_2D_mask = torch.zeros((num_gaussians), dtype=torch.bool).to(device)
    # Check if valid points are within the mask
    points_inside_2D_mask[valid_indices] = tool_mask_soft[rows[valid_indices], cols[valid_indices]]
    # points_inside_2D_mask[valid_indices][points_to_drop] = torch.ones_like(points_inside_2D_mask[valid_indices][points_to_drop]).bool()

    return points_inside_2D_mask#, soft_mask
    

def check_within_2D_mask(samples_xyz_in_cam,tool_mask,K,num_gaussians,
                            dbg_vis_tool_adc = False,
                            current_iter = None,
                            max_iter = None,
                            tool_name = '',
                            tool_adc_mode = '',
                            keep_out_of_frame_pts = True,
                            ):
    '''
    samples_xyz: num,3
    mask: h w
    k: 3*3
    notice can only constrian with init_mask, we can only easily get samples_xyz_in_cam for frist frame consideing current implementation
    '''
    import cv2

    #project xyz on 2D
    assert K.dim() == 2,K.dim()
    assert K.shape[-1] == 3
    assert samples_xyz_in_cam.dim() == 2
    proj = (K @ samples_xyz_in_cam.T).T
    proj_2d = proj[:,:2]/proj[:,2:]
    proj_2d = proj_2d.to(torch.long)
    assert proj_2d.dim() == 2
    assert proj_2d.shape[-1]==2
    assert tool_mask.dim() == 2 ,tool_mask.shape
    assert proj_2d.shape[0] == num_gaussians


    filter_mode = 'hard'
    filter_mode = 'rand_prob'# have trd n progress tow modes
    filter_mode = 'dis_prob'
    # filter_mode = 'prob_dis'
    # filter_mode = 'progressive_prob'

    points_outside_current_frame = torch.zeros((num_gaussians), dtype=torch.bool).to(samples_xyz_in_cam.device)        
    # Check if points are within bounds of the mask
    cols, rows = proj_2d[:, 0], proj_2d[:, 1]
    valid_indices = (rows >= 0) & (rows < tool_mask.shape[0]) & \
                    (cols >= 0) & (cols < tool_mask.shape[1])
    points_outside_current_frame[~valid_indices] = torch.ones_like(points_outside_current_frame[~valid_indices]).bool()

    if filter_mode=='hard':
        # assert 0,f'{rows}{cols}{tool_mask.shape}'
        # Initialize result tensor (default to True)
        # points_inside_2D_mask = torch.ones((num_gaussians), dtype=torch.bool).to(samples_xyz_in_cam.device)
        points_inside_2D_mask = torch.zeros((num_gaussians), dtype=torch.bool).to(samples_xyz_in_cam.device)
        # Check if valid points are within the mask
        points_inside_2D_mask[valid_indices] = tool_mask[rows[valid_indices], cols[valid_indices]]
    # elif filter_mode == 'prob' or filter_mode == 'progressive_prob':
    # elif 'prob' in filter_mode:
    elif filter_mode in ['dis_prob','rand_prob']:
        points_inside_2D_mask = get_soft_mask_indices_prob(tool_mask, proj_2d,
                                                           valid_indices,
                                                           rows=rows,cols=cols,
                                                           num_gaussians=num_gaussians,
                                                           device=samples_xyz_in_cam.device,
                                                             current_iter=current_iter,
                                                             max_iter=max_iter,
                                                             filter_mode=filter_mode)  # Option 4
    else:
        assert 0 
    
    if keep_out_of_frame_pts:
        points_inside_2D_mask = torch.logical_or(points_inside_2D_mask, points_outside_current_frame)
    
    #///////////////////////////            
    if dbg_vis_tool_adc:
        # Convert the binary mask to a uint8 image (required by OpenCV)
        binary_mask_np = (tool_mask.cpu().numpy() * 255).astype(np.uint8)
        # Convert the binary mask to a BGR image for colored visualization
        binary_mask_color = cv2.cvtColor(binary_mask_np, cv2.COLOR_GRAY2BGR)
        # Create a separate overlay for points
        points_overlay = np.zeros_like(binary_mask_color, dtype=np.uint8)
        # Draw points on the overlay
        for point in proj_2d:
            u, v = int(point[0].item()), int(point[1].item())
            cv2.circle(points_overlay, (u, v), radius=2, color=(0, 0, 255), thickness=-1)

        h, w = binary_mask_color.shape[:2]
        # for point in proj_2d[~points_inside_2D_mask]:
        for point in proj_2d[points_inside_2D_mask]:
            u, v = int(point[0].item()), int(point[1].item())
            # Only plot if point is within image boundaries
            if 0 <= u < w and 0 <= v < h:
                cv2.circle(points_overlay, (u, v), radius=1, color=(0, 255, 0), thickness=-1)
        # print(f'///2D CURRENT: tool {tool_name}:////////#inliners #total////////',len(proj_2d[points_inside_2D_mask]),len(proj_2d))
        # Alpha blending
        alpha_mask = 0.3  # Background mask transparency (0-1)
        alpha_points = 0.7  # Points transparency (0-1)
        # Blend the images
        blended_image = cv2.addWeighted(
            binary_mask_color, 
            alpha_mask,
            points_overlay, alpha_points,
            0
        )        
        
        
        
        vis_trd = 4 #40000
        if proj_2d.shape[0]>vis_trd:
            title = f' Tool{tool_name}: {tool_adc_mode} 2D filter{filter_mode}'
            # -- Outlier ratio {1-(float(points_inside_2D_mask.sum()/float(len(proj_2d))))} (Iter {current_iter}/{max_iter})'
            cv2.imshow(
                        title,
                        # f"Projected 2D Points on Binary Mask: {tool_name}", 
                       blended_image)
            cv2.waitKey(1)
            if 0xFF == ord('q'):  # You can replace 'q' with any key you want
                print("Exiting on key press")
                cv2.destroyAllWindows()

    return points_inside_2D_mask 

def vis_torch_img(rendered_image,topic = ''):
    import cv2
    import numpy as np
    rendered_image_vis = rendered_image.detach().cpu().permute(1, 2, 0)
    rendered_image_vis = (rendered_image_vis.numpy()*255).astype(np.uint8)
    rendered_image_vis = cv2.cvtColor(rendered_image_vis, cv2.COLOR_RGB2BGR)
    # Display the image using OpenCV
    # print('**************************************************')
    cv2.imshow(f"{topic}", rendered_image_vis)
    cv2.waitKey(1)
    if 0xFF == ord('q'):  # You can replace 'q' with any key you want
        print("Exiting on key press")
        cv2.destroyAllWindows()
        
def cal_connectivity_from_points(points=None, radius=0.1, K=10, trajectory=None, least_edge_num=3, node_radius=None, mode='nn', GraphK=4, adaptive_weighting=True):
     # input: [Nv,3]
     # output: information of edges
     # ii : [Ne,] the i th vert
     # extend: [Ne,] j th vert is connect to i th vert.
     # nn: ,  [Ne,] the n th neighbour of i th vert is j th vert.
     # K: constrain the nbr area? jj
    Nv = points.shape[0] if points is not None else trajectory.shape[0]
    if trajectory is None:
        if mode == 'floyd':
            dist_mat = geodesic_distance_floyd(points, K=GraphK)
            dist_mat = dist_mat ** 2
            mask = torch.eye(Nv).bool()
            dist_mat[mask] = torch.inf
            nn_dist, nn_idx = dist_mat.sort(dim=1)
            nn_dist, nn_idx = nn_dist[:, :K], nn_idx[:, :K]
        else:
            import pytorch3d.ops
            knn_res = pytorch3d.ops.knn_points(points[None], points[None], None, None, K=K+1)
            # Remove themselves
            nn_dist, nn_idx = knn_res.dists[0, :, 1:], knn_res.idx[0, :, 1:]  # [Nv, K], [Nv, K]
    else:
        trajectory = trajectory.reshape([Nv, -1]) / trajectory.shape[1]  # Average distance of trajectory
        if mode == 'floyd':
            dist_mat = geodesic_distance_floyd(trajectory, K=GraphK)
            dist_mat = dist_mat ** 2
            mask = torch.eye(Nv).bool()
            dist_mat[mask] = torch.inf
            nn_dist, nn_idx = dist_mat.sort(dim=1)
            nn_dist, nn_idx = nn_dist[:, :K], nn_idx[:, :K]
        else:
            knn_res = pytorch3d.ops.knn_points(trajectory[None], trajectory[None], None, None, K=K+1)
            # Remove themselves
            nn_dist, nn_idx = knn_res.dists[0, :, 1:], knn_res.idx[0, :, 1:]  # [Nv, K], [Nv, K]

    # Make sure ranges are within the radius
    nn_idx[:, least_edge_num:] = torch.where(nn_dist[:, least_edge_num:] < radius ** 2, nn_idx[:, least_edge_num:], - torch.ones_like(nn_idx[:, least_edge_num:]))
    
    nn_dist[:, least_edge_num:] = torch.where(nn_dist[:, least_edge_num:] < radius ** 2, nn_dist[:, least_edge_num:], torch.ones_like(nn_dist[:, least_edge_num:]) * torch.inf)
    if adaptive_weighting:
        weight = torch.exp(-nn_dist / nn_dist.mean())
    elif node_radius is None:
        weight = torch.exp(-nn_dist)
    else:
        nn_radius = node_radius[nn_idx]
        weight = torch.exp(-nn_dist / (2 * nn_radius ** 2))
    weight = weight / weight.sum(dim=-1, keepdim=True)

    ii = torch.arange(Nv)[:, None].cuda().long().expand(Nv, K).reshape([-1])
    jj = nn_idx.reshape([-1])
    nn = torch.arange(K)[None].cuda().long().expand(Nv, K).reshape([-1])
    mask = jj != -1
    ii, jj, nn = ii[mask], jj[mask], nn[mask]

    return ii, jj, nn, weight