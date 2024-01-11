import numpy as np
import torch


ROTMATRIX_FEATURE_SIZE = 9
REP6D_FEATURE_SIZE = 6
QUAT_FEATURE_SIZE = 4
EULER_FEATURE_SIZE = 3


def euler_to_rotmatrix(euler: torch.Tensor) -> torch.Tensor:
    """
    Converts batch of euler rotations to batch of rotmatrix
    :param euler: N*3 with roll, pitch, yaw, expressed in radian
    :return: N*3*3
    """
    roll, pitch, yaw = euler[:, 0], euler[:, 1], euler[:, 2]
    c1 = torch.cos(roll)
    s1 = torch.sin(roll)
    c2 = torch.cos(pitch)
    s2 = torch.sin(pitch)
    c3 = torch.cos(yaw)
    s3 = torch.sin(yaw)
    rot_matrices = torch.empty(
        (euler.shape[0], 3, 3), dtype=euler.dtype, device=euler.device
    )
    rot_matrices[:, 0, 0] = c2 * c3
    rot_matrices[:, 0, 1] = -c2 * s3
    rot_matrices[:, 0, 2] = s2
    rot_matrices[:, 1, 0] = c1 * s3 + c3 * s1 * s2
    rot_matrices[:, 1, 1] = c1 * c3 - s1 * s2 * s3
    rot_matrices[:, 1, 2] = -c2 * s1
    rot_matrices[:, 2, 0] = s1 * s3 - c1 * c3 * s2
    rot_matrices[:, 2, 1] = c3 * s1 + c1 * s2 * s3
    rot_matrices[:, 2, 2] = c1 * c2
    return rot_matrices


def rotmatrix_to_euler(rot_matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts batch of rotation matrices to batch of Euler angles.
    Parameters:
    rot_matrix (torch.Tensor): Tensor of shape [N, 3, 3] representing the rotation matrices.
    Returns:
    torch.Tensor: Tensor of shape [N, 3] representing the Euler angles (roll, pitch, yaw).
    """
    euler_angles = torch.empty(
        (rot_matrix.shape[0], 3), dtype=rot_matrix.dtype, device=rot_matrix.device
    )
    theta = torch.asin(rot_matrix[:, 0, 2])
    # Prepare to handle gimbal lock (when theta is +-90 degrees)
    near_pi_over_2 = torch.isclose(
        torch.abs(theta), torch.tensor(torch.pi / 2, device=rot_matrix.device)
    )
    not_near_pi_over_2 = ~near_pi_over_2
    # Roll (phi) and yaw (psi) - first and third angles
    # When NOT in gimbal lock
    euler_angles[not_near_pi_over_2, 0] = torch.atan2(
        -rot_matrix[not_near_pi_over_2, 1, 2], rot_matrix[not_near_pi_over_2, 2, 2]
    )
    euler_angles[not_near_pi_over_2, 2] = torch.atan2(
        -rot_matrix[not_near_pi_over_2, 0, 1], rot_matrix[not_near_pi_over_2, 0, 0]
    )
    # When in gimbal lock, set roll (phi) to zero and compute yaw (psi)
    euler_angles[near_pi_over_2, 0] = 0
    euler_angles[near_pi_over_2, 2] = torch.atan2(
        rot_matrix[near_pi_over_2, 1, 0], rot_matrix[near_pi_over_2, 1, 1]
    )
    euler_angles[:, 1] = theta

    return euler_angles


def quat_to_rotmatrix(quat: torch.Tensor) -> torch.Tensor:
    """
    Converts batch of quaternion rotations to batch of rotmatrix
    :param quat: N*4
    :return: N*3*3
    """
    # Normalize the quaternions
    norm = torch.sqrt(torch.sum(quat**2, dim=1, keepdim=True))
    eps = 1e-6
    # Quaternions with very small norm are set to [0, 0, 0, 1] (no rotation)
    norm_quat = torch.where(
        norm > eps,
        quat / norm,
        torch.tensor([0, 0, 0, 1], dtype=quat.dtype, device=quat.device),
    )
    x, y, z, w = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]
    rot_matrices = torch.zeros(
        (quat.shape[0], 3, 3), dtype=quat.dtype, device=quat.device
    )
    rot_matrices[:, 0, 0] = 1 - 2 * y**2 - 2 * z**2
    rot_matrices[:, 0, 1] = 2 * x * y - 2 * z * w
    rot_matrices[:, 0, 2] = 2 * x * z + 2 * y * w
    rot_matrices[:, 1, 0] = 2 * x * y + 2 * z * w
    rot_matrices[:, 1, 1] = 1 - 2 * x**2 - 2 * z**2
    rot_matrices[:, 1, 2] = 2 * y * z - 2 * x * w
    rot_matrices[:, 2, 0] = 2 * x * z - 2 * y * w
    rot_matrices[:, 2, 1] = 2 * y * z + 2 * x * w
    rot_matrices[:, 2, 2] = 1 - 2 * x**2 - 2 * y**2
    return rot_matrices


def rotmatrix_to_quat(rotmatrix: torch.Tensor) -> torch.Tensor:
    """
    Converts a rotation matrix to quaternion
    :param rotmatrix: N * 3 * 3
    :return: N * 4  with
    """
    rotdiff = rotmatrix - rotmatrix.transpose(1, 2)
    r = torch.zeros_like(rotdiff[:, 0])
    r[:, 0] = -rotdiff[:, 1, 2]
    r[:, 1] = rotdiff[:, 0, 2]
    r[:, 2] = -rotdiff[:, 0, 1]
    r_norm = torch.norm(r, dim=1)
    sintheta = r_norm / 2
    r0 = torch.div(r, r_norm.unsqueeze(1).repeat(1, 3) + 0.00000001)
    t1 = rotmatrix[:, 0, 0]
    t2 = rotmatrix[:, 1, 1]
    t3 = rotmatrix[:, 2, 2]
    costheta = (t1 + t2 + t3 - 1) / 2
    theta = torch.atan2(sintheta, costheta)
    q = torch.zeros(rotmatrix.shape[0], 4).float().to(rotmatrix.device)
    q[:, -1] = torch.cos(theta / 2)
    q[:, :-1] = torch.mul(r0, torch.sin(theta / 2).unsqueeze(1).repeat(1, 3))

    return q


def normalize_with_torch(t_tensor):
    """Normalize along the last dimension."""
    return t_tensor / t_tensor.norm(dim=-1, keepdim=True)


def rep6d_to_rotmatrix(rep6d: torch.Tensor) -> torch.Tensor:
    """
    Converts a batch of rep6d to rotmatrix
    :param rep6d: N * 3 * 2
    :return: N * 3 * 3
    """
    a1 = rep6d[:, :, 0]
    a2 = rep6d[:, :, 1]
    # Normalize a1 to get b1
    b1 = normalize_with_torch(a1)
    # Orthogonalize and normalize a2 to get b2
    a2_dot_b1 = torch.sum(a2 * b1, dim=-1, keepdim=True)
    b2 = a2 - a2_dot_b1 * b1
    b2 = normalize_with_torch(b2)
    # Compute cross product to get b3
    b3 = torch.cross(b1, b2, dim=-1)
    # Stack and reshape to get the rotation matrices
    matrix = torch.stack((b1, b2, b3), dim=-1)
    return matrix


def rotmatrix_to_rep6d(rotmatrix: torch.Tensor) -> torch.Tensor:
    """
    Converts a rotmatrix to rep6d
    :param rotmatrix: N * 3 * 3
    :return: N * 3 * 2
    """
    # remove last dimension on last axis
    rep6d = torch.narrow(rotmatrix, -1, 0, rotmatrix.shape[1] - 1)
    return rep6d


def convert_to_rotmatrix(rotation_tensor: torch.Tensor) -> torch.Tensor:
    """
    Converts a rotation tensor to rotmatrix
    :param rotation_tensor: N * F with F in [3, 4, 6, 9]
    :return: N * 9
    """
    if rotation_tensor.shape[-1] == ROTMATRIX_FEATURE_SIZE:
        return rotation_tensor
    if rotation_tensor.shape[-1] == REP6D_FEATURE_SIZE:
        return rep6d_to_rotmatrix(rotation_tensor.reshape(-1, 3, 2)).reshape(-1, 9)
    if rotation_tensor.shape[-1] == QUAT_FEATURE_SIZE:
        return quat_to_rotmatrix(rotation_tensor).reshape(-1, 9)
    if rotation_tensor.shape[-1] == EULER_FEATURE_SIZE:
        return euler_to_rotmatrix(rotation_tensor).reshape(-1, 9)


def convert_to_rep6d(rotation_tensor: torch.Tensor) -> torch.Tensor:
    """
    Converts a rotmatrix to rep6d
    :param rotation_tensor: N * F with F in [3, 4, 6, 9]
    :return: N * 6
    """
    if rotation_tensor.shape[-1] == REP6D_FEATURE_SIZE:
        return rotation_tensor
    if rotation_tensor.shape[-1] != ROTMATRIX_FEATURE_SIZE:
        rotation_tensor = convert_to_rotmatrix(rotation_tensor)
    rotation_tensor = rotation_tensor.reshape(-1, 3, 3)
    return rotmatrix_to_rep6d(rotation_tensor).reshape(-1, 6)


def convert_to_quat(rotation_tensor: torch.Tensor) -> torch.Tensor:
    """
    Converts a rotmatrix to quat
    :param rotation_tensor: N * F with F in [3, 4, 6, 9]
    :return: N * 4
    """
    if rotation_tensor.shape[-1] == QUAT_FEATURE_SIZE:
        return rotation_tensor
    if rotation_tensor.shape[-1] != ROTMATRIX_FEATURE_SIZE:
        rotation_tensor = convert_to_rotmatrix(rotation_tensor)
    rotation_tensor = rotation_tensor.reshape(-1, 3, 3)
    return rotmatrix_to_quat(rotation_tensor)


def convert_to_euler(rotation_tensor: torch.Tensor) -> torch.Tensor:
    """
    Converts a rotmatrix to euler
    :param rotation_tensor: N * F with F in [3, 4, 6, 9]
    :return: N * 3
    """
    if rotation_tensor.shape[-1] == EULER_FEATURE_SIZE:
        return rotation_tensor
    if rotation_tensor.shape[-1] != ROTMATRIX_FEATURE_SIZE:
        rotation_tensor = convert_to_rotmatrix(rotation_tensor)
    rotation_tensor = rotation_tensor.reshape(-1, 3, 3)
    return rotmatrix_to_euler(rotation_tensor)


def apply_rotation(
    rotmatrices: torch.Tensor, transformation_matrix: torch.Tensor
) -> torch.Tensor:
    """
    Apply rotation upon a batch of rotation matrices
    :param rotmatrices: A batch of rotation matrices with shape (N, 3, 3)
    :param transformation_matrix: The transformation matrix with shape (3, 3)
    :return: A tensor of transformed rotation matrices with the shape (N, 3, 3).
    """
    # Unsqueeze the transformation matrix to match the batch dimension
    # transformation_matrix = transformation_matrix.unsqueeze(0)
    # Multiply the transformation matrix with each rotation matrix in the batch
    transformed_matrices = torch.matmul(transformation_matrix, rotmatrices)
    return transformed_matrices
