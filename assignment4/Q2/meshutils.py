# meshutils.py
import numpy as np

# Open3D is the only backend by default
# pip install open3d==0.17.0
import open3d as o3d


# ---------- helpers ----------
def _to_o3d_mesh(verts, faces):
    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(np.asarray(verts))
    m.triangles = o3d.utility.Vector3iVector(np.asarray(faces))
    m.compute_vertex_normals()
    return m

def _from_o3d_mesh(m):
    return np.asarray(m.vertices), np.asarray(m.triangles)

def _basic_clean(m):
    # common clean-up steps
    m.remove_duplicated_vertices()
    m.remove_duplicated_triangles()
    m.remove_degenerate_triangles()
    m.remove_non_manifold_edges()
    m.remove_unreferenced_vertices()
    m.compute_vertex_normals()
    return m


# ---------- Poisson reconstruction ----------
def poisson_mesh_reconstruction(
    points, normals=None, visualize=False,
    outlier_nb_neighbors=20, outlier_std_ratio=10.0,
    depth=9, density_quantile=0.10
):
    """
    points, normals: (N, 3) np.ndarray
    """
    pcd = o3d.geometry.PointCloud()
    pts = np.asarray(points)
    pcd.points = o3d.utility.Vector3dVector(pts)

    # statistical outlier removal
    pcd, ind = pcd.remove_statistical_outlier(
        nb_neighbors=int(outlier_nb_neighbors), std_ratio=float(outlier_std_ratio)
    )

    # normals
    if normals is None:
        pcd.estimate_normals()
    else:
        nrm = np.asarray(normals)
        if ind is not None:
            nrm = nrm[ind]
        pcd.normals = o3d.utility.Vector3dVector(nrm)

    if visualize:
        o3d.visualization.draw_geometries([pcd], point_show_normal=False)

    # Poisson
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=int(depth)
    )

    # density-based trimming
    densities = np.asarray(densities)
    keep = densities >= np.quantile(densities, float(1.0 - density_quantile))
    mesh.remove_vertices_by_mask(~keep)

    mesh = _basic_clean(mesh)

    if visualize:
        o3d.visualization.draw_geometries([mesh])

    vertices, triangles = _from_o3d_mesh(mesh)
    print(f"[INFO] poisson mesh reconstruction: {pts.shape} --> {vertices.shape} / {triangles.shape}")
    return vertices, triangles


# ---------- decimation ----------
def decimate_mesh(
    verts, faces, target, backend="open3d", remesh=False, optimalplacement=True
):
    """
    target: desired number of faces
    backend is kept for API compatibility; only 'open3d' is supported here
    optimalplacement is unused with Open3D but kept for signature compatibility
    """
    _ori_vert_shape, _ori_face_shape = verts.shape, faces.shape

    if backend != "open3d":
        print("[WARN] backend != 'open3d' requested; falling back to Open3D")

    m = _to_o3d_mesh(verts, faces)

    # quadric decimation to target triangles
    m = m.simplify_quadric_decimation(int(target))
    m = _basic_clean(m)

    # optional smoothing to mimic Meshlab isotropic remeshing feel
    if remesh:
        m = m.filter_smooth_taubin(number_of_iterations=3)
        m = _basic_clean(m)

    verts_out, faces_out = _from_o3d_mesh(m)
    print(f"[INFO] mesh decimation: {_ori_vert_shape} --> {verts_out.shape}, {_ori_face_shape} --> {faces_out.shape}")
    return verts_out, faces_out


# ---------- cleaning ----------
def clean_mesh(
    verts, faces, v_pct=1, min_f=8, min_d=5, repair=True, remesh=True, remesh_size=0.01
):
    """
    v_pct: merge-close-vertices strength as percentage of bbox diagonal over 1e4
           matches your comment: 1 means ~diag/10000
    min_f: remove connected components with fewer than min_f faces
    min_d: remove components with diameter < min_d % of whole-mesh bbox diagonal
    repair: run non-manifold and duplicate fixes
    remesh: light smoothing to regularize triangles
    remesh_size: kept for API compatibility (Open3D does not use an explicit target edge here)
    """
    _ori_vert_shape, _ori_face_shape = verts.shape, faces.shape

    m = _to_o3d_mesh(verts, faces)

    # remove unreferenced etc. early
    m = _basic_clean(m)

    # merge close vertices via vertex clustering
    if v_pct and v_pct > 0:
        bbox = m.get_axis_aligned_bounding_box()
        diag = np.linalg.norm(bbox.get_extent())
        voxel = max(1e-12, float(diag) * (float(v_pct) / 10000.0))
        m = m.simplify_vertex_clustering(
            voxel_size=voxel,
            contraction=o3d.geometry.SimplificationContraction.Average
        )
        m = _basic_clean(m)

    # connected components, remove small clusters by face count and diameter
    if min_f > 0 or min_d > 0:
        tri_clusters, cluster_n_triangles, _ = m.cluster_connected_triangles()
        tri_clusters = np.asarray(tri_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)

        # build per-cluster bbox diagonal
        cluster_masks = []
        if min_d > 0:
            overall_bbox = m.get_axis_aligned_bounding_box()
            overall_diag = np.linalg.norm(overall_bbox.get_extent())
            thresh = overall_diag * (float(min_d) / 100.0)

            # compute per-cluster bbox quickly
            verts_np = np.asarray(m.vertices)
            tris_np = np.asarray(m.triangles)

            # init per-cluster extrema
            max_cluster_id = int(tri_clusters.max()) if len(tri_clusters) else -1
            cluster_min = np.full((max_cluster_id + 1, 3), np.inf, dtype=np.float64)
            cluster_max = np.full((max_cluster_id + 1, 3), -np.inf, dtype=np.float64)

            for t_idx, c in enumerate(tri_clusters):
                tri = tris_np[t_idx]
                v = verts_np[tri]
                cluster_min[c] = np.minimum(cluster_min[c], v.min(axis=0))
                cluster_max[c] = np.maximum(cluster_max[c], v.max(axis=0))

            cluster_diag = np.linalg.norm(cluster_max - cluster_min, axis=1)
        else:
            cluster_diag = None
            thresh = None

        remove_mask = np.zeros(len(m.triangles), dtype=bool)
        for c_id, ntri in enumerate(cluster_n_triangles):
            small_by_faces = (min_f > 0 and ntri < int(min_f))
            small_by_diam = (min_d > 0 and cluster_diag is not None and cluster_diag[c_id] < thresh)
            if small_by_faces or small_by_diam:
                remove_mask |= (tri_clusters == c_id)

        if remove_mask.any():
            m.remove_triangles_by_mask(remove_mask)
            m.remove_unreferenced_vertices()
            m = _basic_clean(m)

    # repair-like steps (already covered by _basic_clean, kept for API parity)
    if repair:
        m = _basic_clean(m)

    # light smoothing to emulate Meshlab isotropic remeshing feel
    if remesh:
        m = m.filter_smooth_taubin(number_of_iterations=3)
        m = _basic_clean(m)

    verts_out, faces_out = _from_o3d_mesh(m)
    print(f"[INFO] mesh cleaning: {_ori_vert_shape} --> {verts_out.shape}, {_ori_face_shape} --> {faces_out.shape}")
    return verts_out, faces_out