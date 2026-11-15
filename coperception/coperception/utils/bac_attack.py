"""
BAC (Blind Area Confusion) Attack Implementation
Based on paper: GCP - Guarded Collaborative Perception

Implements:
1. Differential Detection: Standalone-victim and collaborative-only detections
2. Blind Region Segmentation (BRS): Algorithm 1 from paper
3. BAC Perturbation Optimization: Equation 4 from paper

Reconciliation policy (user directive 2026-11-11): every design point is
classified as paper-explicit (implemented literally from the paper, even when
the author snapshot differs) or paper-silent (the author snapshot's open-source
behavior is used verbatim). The full decision table lives in
generate_bac_attack's docstring and audit/patches/PATCHES.md.
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
from collections import deque
from scipy.optimize import linear_sum_assignment

# Global cache for BAC masks (slow update strategy)
_bac_mask_cache = {}
_bac_mask_frame_counter = {}


def extract_detection_boxes_from_result(result, config, padded_voxel_points,
                                         reg_target, anchors_map, gt_max_iou, agent_idx):
    """Author open-source extraction (paper does not define how detections for
    the differential analysis are obtained): cal_local_mAP with the repository
    score thresholds, graceful empty result on failure."""
    try:
        from coperception.utils.detection_util import cal_local_mAP

        # Check if result is valid
        if result is None or agent_idx >= len(result):
            return np.array([])

        result_temp = result[agent_idx]
        if result_temp is None or len(result_temp) == 0:
            return np.array([])

        # Construct data dictionary (same format as local_eval)
        data_agents = {
            "bev_seq": torch.unsqueeze(padded_voxel_points[agent_idx, :, :, :, :], 1),
            "reg_targets": torch.unsqueeze(reg_target[agent_idx, :, :, :, :, :], 0),
            "anchors": torch.unsqueeze(anchors_map[agent_idx, :, :, :, :], 0),
        }

        # Process gt_max_iou
        temp_gt = gt_max_iou[agent_idx]
        if len(temp_gt[0]["gt_box"]) == 0:
            data_agents["gt_max_iou"] = []
        else:
            data_agents["gt_max_iou"] = temp_gt[0]["gt_box"][0, :, :]

        # Construct temp dictionary for cal_local_mAP
        temp = {
            "bev_seq": data_agents["bev_seq"][0, -1].cpu().numpy(),
            "result": [] if len(result_temp) == 0 else result_temp[0][0],
            "reg_targets": data_agents["reg_targets"].cpu().numpy()[0],
            "anchors_map": data_agents["anchors"].cpu().numpy()[0],
            "gt_max_iou": data_agents["gt_max_iou"],
        }

        # Call cal_local_mAP to extract detection boxes
        det_results, _ = cal_local_mAP(config, temp, [], [])

        # Return detection boxes
        if len(det_results) > 0 and len(det_results[0]) > 0:
            boxes = det_results[0][0]  # Shape: (N, 9)
            return boxes

        return np.array([])

    except Exception as e:
        print(f"[extract_detection_boxes] Error: {e}")
        import traceback
        traceback.print_exc()
        return np.array([])



def normalize_box_array(boxes):
    """Return detection boxes as a 2D numpy array with shape [N, >=8]."""
    if boxes is None:
        return np.empty((0, 9), dtype=np.float32)

    boxes = np.asarray(boxes)
    if boxes.size == 0:
        return np.empty((0, 9), dtype=np.float32)

    if boxes.ndim == 1:
        boxes = boxes.reshape(1, -1)

    return boxes.astype(np.float32, copy=False)


def calculate_iou_from_boxes(box1, box2):
    """Calculate IoU between two detection boxes represented by 4 BEV corners."""
    from shapely.geometry import Polygon

    box1 = np.asarray(box1).reshape(-1)
    box2 = np.asarray(box2).reshape(-1)
    if box1.shape[0] < 8 or box2.shape[0] < 8:
        return 0.0

    poly1 = Polygon(box1[:8].reshape(4, 2))
    poly2 = Polygon(box2[:8].reshape(4, 2))
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0

    union = poly1.area + poly2.area - poly1.intersection(poly2).area
    if union <= 0:
        return 0.0

    return poly1.intersection(poly2).area / union


def differential_detection(single_boxes, cp_boxes, iou_threshold=0.3):
    """
    Partition detections into matched, victim-only, and collaborative-only groups.

    `single_boxes` correspond to the victim's standalone perception.
    `cp_boxes` correspond to collaborative perception with the malicious sender.
    """
    single_boxes = normalize_box_array(single_boxes)
    cp_boxes = normalize_box_array(cp_boxes)

    if len(single_boxes) == 0 and len(cp_boxes) == 0:
        empty = np.empty((0, 9), dtype=np.float32)
        return empty, empty, empty, empty
    if len(single_boxes) == 0:
        empty = np.empty((0, cp_boxes.shape[1]), dtype=np.float32)
        return empty, cp_boxes, empty, empty
    if len(cp_boxes) == 0:
        empty = np.empty((0, single_boxes.shape[1]), dtype=np.float32)
        return single_boxes, empty, empty, empty

    iou_matrix = np.zeros((len(single_boxes), len(cp_boxes)), dtype=np.float32)
    for i, s_box in enumerate(single_boxes):
        for j, c_box in enumerate(cp_boxes):
            iou_matrix[i, j] = calculate_iou_from_boxes(s_box, c_box)

    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    matched_pairs = [
        (row_idx, col_idx)
        for row_idx, col_idx in zip(row_ind, col_ind)
        if iou_matrix[row_idx, col_idx] >= iou_threshold
    ]

    matched_single_idx = {row_idx for row_idx, _ in matched_pairs}
    matched_cp_idx = {col_idx for _, col_idx in matched_pairs}

    victim_only = single_boxes[
        [idx for idx in range(len(single_boxes)) if idx not in matched_single_idx]
    ]
    collaborative_only = cp_boxes[
        [idx for idx in range(len(cp_boxes)) if idx not in matched_cp_idx]
    ]
    matched_single = single_boxes[sorted(matched_single_idx)] if matched_single_idx else np.empty((0, single_boxes.shape[1]), dtype=np.float32)
    matched_cp = cp_boxes[sorted(matched_cp_idx)] if matched_cp_idx else np.empty((0, cp_boxes.shape[1]), dtype=np.float32)

    return victim_only, collaborative_only, matched_single, matched_cp


def box_to_grid(box, grid_size, map_range=(-32, 32)):
    """Author open-source rasterization (paper does not specify rotated-box
    rasterization or out-of-range rules): clip the box's axis-aligned extent;
    degenerate/invalid boxes fall back to the grid centre."""
    try:
        H, W = grid_size
        min_coord, max_coord = map_range
        map_size = max_coord - min_coord  # 64米

        # Ensure box is numpy array
        if not isinstance(box, np.ndarray):
            box = np.array(box)

        # Extract 4 corners
        box_flat = box.flatten()  # Ensure 1D
        if box_flat.shape[0] < 8:
            return [(H//2, W//2)]

        corners = box_flat[:8].reshape(4, 2)  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

        # Calculate bounding box
        x_min, y_min = float(corners[:, 0].min()), float(corners[:, 1].min())
        x_max, y_max = float(corners[:, 0].max()), float(corners[:, 1].max())

        # Convert world coordinates to grid coordinates [-32, 32] -> [0, H-1]
        grid_x_min = int(np.clip((x_min - min_coord) / map_size * W, 0, W-1))
        grid_x_max = int(np.clip((x_max - min_coord) / map_size * W, 0, W-1))
        grid_y_min = int(np.clip((y_min - min_coord) / map_size * H, 0, H-1))
        grid_y_max = int(np.clip((y_max - min_coord) / map_size * H, 0, H-1))

        # Generate grid coordinates list
        coords = []
        for i in range(grid_y_min, grid_y_max + 1):
            for j in range(grid_x_min, grid_x_max + 1):
                if 0 <= i < H and 0 <= j < W:
                    coords.append((i, j))

        return coords if len(coords) > 0 else [(H//2, W//2)]

    except Exception as e:
        print(f"[box_to_grid] Error: {e}, returning center")
        H, W = grid_size
        return [(H//2, W//2)]



def cluster_victim_grid(Y_vic, grid_size=(32, 32), map_range=(-32, 32)):
    """
    Find the victim grid e = the grid point minimizing the total distance to
    all victim detection boxes (paper Algorithm 1, Line 2: 'e <- Cluster(Y_vic)
    // Victim Grid'; Sec. III-C: 'finding the grid point with the minimum total
    distance to all victim-detected objects').

    [repro-patch P11a] the author snapshot MAXIMIZED this total distance,
    which placed the region-growing center at the periphery instead of the
    perception-centric position defined by the paper.

    Args:
        Y_vic: Victim detection boxes (N, 9)
        grid_size: BEV grid size (H, W)
        map_range: BEV map range

    Returns:
        e: Grid coordinates of victim center (i, j)
    """
    if not isinstance(Y_vic, np.ndarray) or Y_vic.ndim != 2 or Y_vic.shape[0] == 0:
        return (grid_size[0] // 2, grid_size[1] // 2)

    H, W = grid_size
    min_coord, max_coord = map_range
    map_size = max_coord - min_coord

    # Convert all detection box centers to grid coordinates
    box_centers_grid = []
    for box in Y_vic:
        corners = box[:8].reshape(4, 2)
        cx, cy = corners.mean(axis=0)
        # Convert world coordinates to grid coordinates
        gi = int(np.clip((cy - min_coord) / map_size * H, 0, H-1))
        gj = int(np.clip((cx - min_coord) / map_size * W, 0, W-1))
        box_centers_grid.append((gi, gj))

    # [P11a] minimize total distance to all victim detections (paper)
    min_dist = float('inf')
    victim_grid = (H // 2, W // 2)

    for i in range(H):
        for j in range(W):
            total_dist = 0
            for gi, gj in box_centers_grid:
                dist = np.sqrt((i - gi)**2 + (j - gj)**2)
                total_dist += dist

            if total_dist < min_dist:
                min_dist = total_dist
                victim_grid = (i, j)

    return victim_grid


def get_adaptive_neighbors(s, e, K_base=6, gamma_d=0.3, D_norm=45.25):
    """
    Adaptive neighborhood selection (Equation 3 in paper)
    K_s = K_base * exp(-gamma_d * d(s,e) / D_norm)
    
    Args:
        s: Current grid point (i, j)
        e: Victim grid center (i, j)
        K_base: Base number of neighbors (default 6)
        gamma_d: Decay rate (default 0.3)
        D_norm: Normalization factor sqrt(H^2 + W^2)
        
    Returns:
        K_s: Number of neighbors for grid s
    """
    dist = np.sqrt((s[0] - e[0])**2 + (s[1] - e[1])**2)
    K_s = int(K_base * np.exp(-gamma_d * dist / D_norm))
    return max(K_s, 1)  # At least 1 neighbor


def get_neighbors(point, H, W, K=6):
    """
    Repository choice: 8-connected stencil with deterministic ordering.
    The paper specifies K_s but not the stencil or tie-breaking order.
    
    Args:
        point: Current grid point (i, j)
        H, W: Grid dimensions
        K: Number of neighbors to return
        
    Returns:
        neighbors: List of neighbor coordinates
    """
    i, j = point
    
    # 8-connected neighbors
    all_neighbors = [
        (i-1, j-1), (i-1, j), (i-1, j+1),
        (i, j-1),             (i, j+1),
        (i+1, j-1), (i+1, j), (i+1, j+1)
    ]
    
    # Filter valid neighbors
    valid_neighbors = []
    for ni, nj in all_neighbors:
        if 0 <= ni < H and 0 <= nj < W:
            valid_neighbors.append((ni, nj))
    
    # Keep repository ordering; this ordering is NOT specified by the paper.
    return valid_neighbors[:min(K, len(valid_neighbors))]


def blind_region_segmentation(detection_map, Y_vic, Y_nvic, grid_size=(32, 32),
                               K_base=6, gamma_d=0.3, map_range=(-32, 32),
                               confidence_threshold=0.3):
    """Paper Algorithm 1: seed labels, alternate queue growth, BA -> 0.

    Y_vic is ALL standalone-victim detections (Sec. III-C: Y_vic = Y_s).
    The paper specifies no confidence-intensity threshold for region growth;
    detection_map supplies the lattice only. confidence_threshold is an unused
    compatibility argument. Grid resolution, box rasterization, neighbor order,
    and tie breaking are repository choices, not specified by the paper.
    """
    H, W = grid_size
    if H <= 0 or W <= 0:
        raise ValueError("BRS grid dimensions must be positive")
    Y_vic, Y_nvic = normalize_box_array(Y_vic), normalize_box_array(Y_nvic)
    M_i = np.zeros((H, W), dtype=np.float32)
    e = cluster_victim_grid(Y_vic, grid_size, map_range)
    for box in Y_vic:
        for i, j in box_to_grid(box, grid_size, map_range):
            M_i[i, j] = 1

    # Paper-silent fallback (author open-source code): no victim detections ->
    # a 3x3 confident seed block at the grid centre.
    if not np.any(M_i == 1):
        center_i, center_j = H // 2, W // 2
        for di in range(-1, 2):
            for dj in range(-1, 2):
                i, j = center_i + di, center_j + dj
                if 0 <= i < H and 0 <= j < W:
                    M_i[i, j] = 1

    # Algorithm 1 lines 3-4: empty Y_nvic -> farthest grid from e.
    if len(Y_nvic) == 0:
        rows, cols = np.indices((H, W))
        distance_sq = (rows - e[0]) ** 2 + (cols - e[1]) ** 2
        seed = np.unravel_index(np.argmax(distance_sq), (H, W))
        M_i[seed] = -1
    else:
        # Line 6 assigns BA after CA. On overlap BA therefore takes precedence.
        for box in Y_nvic:
            for i, j in box_to_grid(box, grid_size, map_range):
                M_i[i, j] = -1

    # Build queues AFTER all assignments (line 7); no fabricated center CA seeds.
    Q_ca = deque(map(tuple, np.argwhere(M_i == 1)))
    Q_ba = deque(map(tuple, np.argwhere(M_i == -1)))
    print("[BRS] Victim grid: %s; CA seeds=%d, BA seeds=%d" %
          (e, len(Q_ca), len(Q_ba)))
    D_norm = np.sqrt(H ** 2 + W ** 2)
    while Q_ca or Q_ba:
        for queue, label in ((Q_ca, 1), (Q_ba, -1)):
            if not queue:
                continue
            point = queue.popleft()
            K_s = get_adaptive_neighbors(point, e, K_base, gamma_d, D_norm)
            for i, j in get_neighbors(point, H, W, K_s):
                if M_i[i, j] == 0:
                    M_i[i, j] = label
                    queue.append((i, j))
    # Algorithm 1 line 17: negative labels become zero; unassigned stay zero.
    return (M_i == 1).astype(np.float32)



def bac_mask_update_interval(frame_rate_hz):
    """Convert a known simulation sampling rate to the paper's 0.5 Hz cadence."""
    if (frame_rate_hz is None or not np.isfinite(frame_rate_hz) or
            frame_rate_hz <= 0):
        raise ValueError("BAC requires the actual data frame rate: --frame_rate_hz")
    return max(1, int(np.ceil(float(frame_rate_hz) / 0.5)))


def _bac_model_output(fafmodule, data, pert, attacker_list, eps, ego_agent):
    """Use the same differentiable inference path as FaFModule.predict_all."""
    config = getattr(fafmodule, 'config', None)
    flag = getattr(config, 'flag', '')
    kwargs = dict(batch_size=1, pert=pert, no_fuse=False, unadv_pert=None,
                  attacker_list=attacker_list, eps=eps, ego_agent=ego_agent)
    if flag.startswith(('when2com', 'who2com')):
        inference = getattr(config, 'inference', None)
        if inference != 'softmax':
            # activated/argmax paths detach fused features in this repository.
            # Do not silently optimize a different training-mode surrogate.
            raise ValueError("BAC supports When2com's differentiable "
                             "--inference softmax path only")
        kwargs.update(training=False, inference=inference)
    output = fafmodule.model(data['bev_seq'], data['trans_matrices'],
                             data['num_agent'], **kwargs)
    # DiscoNet returns (result, weights); KD variants put result first as well.
    if isinstance(output, (tuple, list)):
        output = output[0]
    if not isinstance(output, dict) or 'cls' not in output:
        raise TypeError("BAC model must return a detection dict (possibly in a tuple)")
    return output


def bac_objective(Y_logits, Y_gt, M_confident, output_eps=0.5, w_delta=1.0):
    """Eq.(4) argmax_δ L(Y_δ⊙W_δ, Y_gt⊙W_δ) with the repository's loss form.

    Paper-specified: the weights W_δ = w_δ·M_δ + S_δ, the target Y_gt
    (ground-truth one-hot labels), and the maximization. Paper-silent
    (author open-source code decides): L is BCE-with-logits on the raw
    logits, written in the author's minimize-against-inverted-labels form --
    for one-hot targets this is sign-equivalent to ascending BCE toward Y_gt.
    S_delta deviations are measured on sigmoid probabilities (Delta_o bounds
    an output-probability shift). The BRS mask enters on the anchor grid via
    the caller's expansion (adapter: the paper leaves the anchor/mask mapping
    unspecified and the author snapshot had no weighted loss at all).
    """
    if Y_logits.shape != Y_gt.shape:
        raise ValueError("BAC output/GT shapes differ")
    if M_confident.shape != Y_logits.shape[:-1]:
        raise ValueError("BAC confidence mask must have one value per anchor")
    probs = torch.sigmoid(Y_logits)
    M_delta = 1.0 - M_confident.unsqueeze(-1)
    S_delta = 1.0 - torch.sigmoid((probs - Y_gt).abs() - output_eps)
    W_delta = w_delta * M_delta + S_delta
    bce_map = F.binary_cross_entropy_with_logits(
        Y_logits, 1.0 - Y_gt, reduction='none'
    )  # [N, A, 2]; minimized-against-inverted == maximized-against-GT
    return (bce_map * W_delta).mean()


def generate_bac_attack(fafmodule, data, num_agent, ego_agent=1,
                        attacker_list=None, num_iterations=10,
                        alpha=0.1, eps=0.5, device='cuda',
                        config=None, reg_target=None, anchors_map=None,
                        gt_max_iou=None, padded_voxel_points=None,
                        scene_id=0, frame_id=0, mask_update_rate=None,
                        output_eps=0.5, w_delta=1.0, frame_rate_hz=None,
                        colluding=False):
    """BRS + Eq.(4), maximizing the victim detection loss against actual GT.

    Reconciliation policy (per user directive 2026-11-11): where the paper
    states a requirement, this implementation follows the paper even when the
    author snapshot differs; where the paper is silent, the author snapshot's
    open-source behavior is used verbatim.

    Paper-explicit (kept from the rewrite): Eq.(4) weights W_delta =
    w_delta*M_delta + S_delta with element-wise S_delta and the GT one-hot
    target Y_gt, maximization (gradient ascent), Eq.(3) K_base=6/gamma_d=0.3,
    Algorithm 1 seeding/growth/binary output, and the paper's stated 0.5 FPS
    mask-update cadence ("we set the mask update rate to 0.5 FPS").

    Paper-silent (author open-source behavior restored): the loss L is
    BCE-with-logits in the author's minimize-against-inverted-labels form
    (sign-equivalent to ascending toward the one-hot Y_gt); box extraction
    via cal_local_mAP with graceful empty results; box->grid rasterization
    with centre fallback; K_s truncation via int(); centre 3x3 CA fallback
    when the victim has no detections.

    Experiment settings (Table I / protocol, CLI-driven): iterations=10,
    Delta_i=eps=0.5, Delta_o=output_eps=0.5, w_delta=1 (paper gives no value),
    m=2, attack_ratio=0.6, seed=42, frame_rate_hz=10.

    Additional adapter choices (no author counterpart exists): the anchor-grid
    expansion of M_delta, classification-only optimization, joint m-attacker
    optimization, and the random small initialization.

    Returns the existing (perturbation, binary_confidence_mask) tuple.
    Does not update model parameters or mutate data; errors propagate.
    """
    if attacker_list is None:
        raise ValueError("BAC requires an explicit attacker_list")
    attacker_list = list(attacker_list)
    if len(set(attacker_list)) != len(attacker_list):
        raise ValueError("Duplicate BAC attackers")
    if not 0 <= ego_agent < num_agent or any(
            not 0 <= i < num_agent or i == ego_agent for i in attacker_list):
        raise ValueError("Invalid BAC victim/attacker indices")
    if (not all(np.isfinite(v) for v in (eps, alpha, output_eps, w_delta)) or
            eps < 0 or alpha < 0 or output_eps < 0 or w_delta <= 0 or
            num_iterations < 0):
        raise ValueError("Invalid BAC budgets, weight, or iteration count")
    if anchors_map is None or 'labels' not in data:
        raise ValueError("Paper Eq.(4) requires anchors and actual GT labels")
    if frame_rate_hz is None:
        frame_rate_hz = data.get('frame_rate_hz')
    interval = bac_mask_update_interval(frame_rate_hz)
    if mask_update_rate is not None and mask_update_rate != interval:
        raise ValueError("mask_update_rate conflicts with 0.5 Hz at the supplied frame rate")
    mask_update_rate = interval
    config_flag = getattr(getattr(fafmodule, 'config', None), 'flag', '')
    if (config_flag.startswith(('when2com', 'who2com')) and
            getattr(fafmodule.config, 'inference', None) != 'softmax'):
        raise ValueError("BAC requires --inference softmax for When2com")

    # Feature shape follows the selected repository fusion layer, including
    # layer 4 (512x16x16); do not silently force the old layer-3 shape.
    model = fafmodule.model.module if hasattr(fafmodule.model, 'module') else fafmodule.model
    layer = model.layer
    feature_shapes = ((32, 256, 256), (64, 128, 128), (128, 64, 64),
                      (256, 32, 32), (512, 16, 16))
    if layer not in range(len(feature_shapes)):
        raise ValueError("Unsupported repository fusion layer: %s" % layer)
    C, Hf, Wf = feature_shapes[layer]
    Hm, Wm, anchor_num = map(int, anchors_map.shape[1:4])
    grid = int(os.environ.get('BAC_VIS_GRID', str(Hf)))
    cache_key = (scene_id, ego_agent, tuple(attacker_list), num_agent, grid, Hm, Wm,
                 float(frame_rate_hz))
    last_frame = _bac_mask_frame_counter.get(cache_key)
    update_tick = int(np.floor(float(frame_id) * 0.5 / float(frame_rate_hz)))
    last_tick = (int(np.floor(float(last_frame) * 0.5 / float(frame_rate_hz)))
                 if last_frame is not None else None)
    if (cache_key not in _bac_mask_cache or last_frame is None or
            frame_id < last_frame or update_tick != last_tick):
        print("[BAC] Mask cadence: 0.5 Hz on %s Hz simulation data "
              "(at most %s frames per update bucket)" % (frame_rate_hz, mask_update_rate))
        mask = _generate_bac_mask(
            fafmodule, data, num_agent, ego_agent, attacker_list, device,
            config, reg_target, anchors_map, gt_max_iou, padded_voxel_points,
            scene_id=scene_id, frame_id=frame_id, grid_size=(grid, grid))
        _bac_mask_cache[cache_key] = mask.copy()
        _bac_mask_frame_counter[cache_key] = frame_id
    else:
        mask = _bac_mask_cache[cache_key].copy()

    dtype = data['bev_seq'].dtype
    M_confident = torch.as_tensor(mask, dtype=dtype, device=device)
    M_cell = F.interpolate(M_confident.unsqueeze(1), size=(Hm, Wm), mode='nearest')
    M_anchor = (M_cell.squeeze(1).reshape(num_agent, Hm * Wm, 1)
                .expand(num_agent, Hm * Wm, anchor_num)
                .reshape(num_agent, Hm * Wm * anchor_num))
    labels = data['labels'].detach().to(device=device, dtype=dtype)
    labels = labels.reshape(num_agent, Hm * Wm * anchor_num, -1)
    if not torch.isfinite(labels).all():
        raise ValueError("Non-finite BAC GT labels")
    # Inactive padded agents legitimately have all-zero labels. Only the
    # attacked victim is used in the objective and must have valid one-hot GT.
    victim_labels = labels[ego_agent]
    if (not torch.all((victim_labels == 0) | (victim_labels == 1)) or
            not torch.all(victim_labels.sum(dim=-1) == 1)):
        raise ValueError("BAC victim classification GT must be one-hot")

    # Keep the old small random start as an explicitly non-paper-specified choice.
    active = torch.zeros((num_agent, 1, 1, 1), dtype=dtype, device=device)
    active[attacker_list] = 1
    pert = (torch.randn(num_agent, C, Hf, Wf, dtype=dtype, device=device)
            * 0.01).clamp(-eps, eps) * active
    if not attacker_list:
        return pert.detach(), mask
    victim_slice = slice(ego_agent, ego_agent + 1)
    for iteration in range(num_iterations):
        pert = pert.detach().requires_grad_(True)
        result = _bac_model_output(fafmodule, data, pert, attacker_list, eps, ego_agent)
        Y_logits = result['cls'][victim_slice]
        Y_gt = labels[victim_slice]
        objective = bac_objective(Y_logits, Y_gt, M_anchor[victim_slice],
                                  output_eps=output_eps, w_delta=w_delta)
        if not torch.isfinite(objective):
            raise RuntimeError("Non-finite BAC objective at iteration %d" % iteration)
        # Request only delta's gradient: no optimizer.step or model .grad writes.
        grad, = torch.autograd.grad(objective, pert)
        if not torch.isfinite(grad).all():
            raise RuntimeError("Non-finite BAC gradient at iteration %d" % iteration)
        with torch.no_grad():
            # [E02] Colluding mode: alternating block-coordinate updates --
            # one attacker's block per iteration, the other attacker's current
            # perturbation stays active in the fusion (joint objective).
            if colluding:
                keep = attacker_list[iteration % len(attacker_list)]
                for idx in attacker_list:
                    if idx != keep:
                        grad[idx] = 0
            pert = (pert + alpha * grad.sign()).clamp(-eps, eps) * active
    print("[BAC] Optimization completed, pert norm: %.4f" % pert.detach().norm().item())
    return pert.detach(), mask



def _generate_bac_mask(fafmodule, data, num_agent, ego_agent, attacker_list, device,
                       config, reg_target, anchors_map, gt_max_iou, padded_voxel_points,
                       scene_id=0, frame_id=0, grid_size=(32, 32)):
    """
    Internal function to generate BAC mask (slow update)
    """
    print("[BAC] Step 1: Differential Detection...")
    
    # Get victim-only result
    data_vic = {k: v for k, v in data.items()}
    data_vic['pert'] = None
    data_vic['no_fuse'] = True
    data_vic['collab_agent_list'] = None
    data_vic['trial_agent_id'] = None
    data_vic['ego_agent'] = ego_agent
    result_vic = fafmodule.predict_all(data_vic, 1, validation=False, num_agent=num_agent)
    
    # Get CP result (with one collaborator)
    data_cp = {k: v for k, v in data.items()}
    data_cp['pert'] = None
    data_cp['collab_agent_list'] = [attacker_list[0]] if len(attacker_list) > 0 else None
    data_cp['no_fuse'] = False
    data_cp['trial_agent_id'] = None
    data_cp['ego_agent'] = ego_agent
    result_cp = fafmodule.predict_all(data_cp, 1, validation=False, num_agent=num_agent)
    
    # Extract detection boxes from standalone victim perception and collaborative perception.
    Y_single_boxes_raw = extract_detection_boxes_from_result(
        result_vic, config, padded_voxel_points, 
        reg_target, anchors_map, gt_max_iou, ego_agent
    )
    
    Y_cp_boxes_raw = extract_detection_boxes_from_result(
        result_cp, config, padded_voxel_points,
        reg_target, anchors_map, gt_max_iou, ego_agent
    )

    Y_single_boxes = normalize_box_array(Y_single_boxes_raw)
    Y_cp_boxes = normalize_box_array(Y_cp_boxes_raw)
    Y_vic_boxes, Y_nvic_boxes, Y_matched_single, Y_matched_cp = differential_detection(
        Y_single_boxes, Y_cp_boxes, iou_threshold=0.3
    )

    print(
        "[BAC] Differential detection: "
        f"single={len(Y_single_boxes)}, cp={len(Y_cp_boxes)}, "
        f"matched={len(Y_matched_single)}, victim_only={len(Y_vic_boxes)}, "
        f"collaborative_only={len(Y_nvic_boxes)}"
    )
    
    # Step 2: Blind Region Segmentation
    print("[BAC] Step 2: Blind Region Segmentation...")
    
    # Algorithm 1 uses a detection-map lattice, not intensity-threshold growth.
    # Lattice resolution and rasterization are explicitly repository choices.
    detection_map = np.zeros(grid_size, dtype=np.float32)
    blind_mask = blind_region_segmentation(
        detection_map, Y_single_boxes, Y_nvic_boxes,
        grid_size=grid_size, K_base=6, gamma_d=0.3)

    # Expand to all agents
    blind_mask_full = np.tile(blind_mask, (num_agent, 1, 1))

    blind_ratio = 1.0 - blind_mask.mean()
    confident_ratio = blind_mask.mean()

    # [P11c] removed the snapshot's 'extreme blind ratio -> balanced
    # half-split' replacement: Algorithm 1 has no such mechanism, and the
    # replacement produced a non-binary mask. The genuine BRS output (binary)
    # is always returned; extreme ratios are reported for the record only.
    if blind_ratio > 0.95 or blind_ratio < 0.05:
        print(f"[BAC] Note: extreme blind ratio ({blind_ratio:.2%}); "
              f"kept Algorithm 1 binary output")
        confident_ratio = blind_mask.mean()
    
    print(f"[BAC] Generated blind mask: blind={blind_ratio:.2%}, confident={confident_ratio:.2%}")

    # [repro-patch P9] optional export of BAC intermediates for Fig.7 style
    # visualization (victim-only / victim+malicious / seed map / BRS mask).
    # Pure side-effect: nothing below depends on this; disabled unless the
    # BAC_DEBUG_DIR environment variable is set.
    _dbg = os.environ.get('BAC_DEBUG_DIR')
    if _dbg:
        try:
            os.makedirs(_dbg, exist_ok=True)
            # [P12] unique per update: the old stamp ('last' counter key was
            # never written) collapsed every export of a run onto one file.
            stamp = f'scene{scene_id}_frame{int(frame_id):03d}'
            np.savez(os.path.join(_dbg, f'bac_mask_intermediate_{stamp}.npz'),
                     Y_vic_boxes=np.array(Y_vic_boxes) if len(Y_vic_boxes) else np.zeros((0, 8)),
                     Y_nvic_boxes=np.array(Y_nvic_boxes) if len(Y_nvic_boxes) else np.zeros((0, 8)),
                     Y_matched_single=np.array(Y_matched_single) if len(Y_matched_single) else np.zeros((0, 8)),
                     Y_matched_cp=np.array(Y_matched_cp) if len(Y_matched_cp) else np.zeros((0, 8)),
                     detection_map=detection_map,
                     blind_mask=blind_mask,
                     scene_id=np.array(scene_id),
                     frame_id=np.array(frame_id))
            print(f"[BAC] Intermediates exported to {_dbg}")
        except Exception as _e:
            print(f"[BAC] Debug export failed: {_e}")
    
    return blind_mask_full
