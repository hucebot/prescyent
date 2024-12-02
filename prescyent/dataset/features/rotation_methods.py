import torch

from prescyent.dataset.features.feature.rotation import (
    Rotation,
    RotationEuler,
    RotationQuat,
    RotationRotMat,
    RotationRep6D,
)

ROTMATRIX_FEATURE_SIZE = 9
REP6D_FEATURE_SIZE = 6
QUAT_FEATURE_SIZE = 4
EULER_FEATURE_SIZE = 3


def euler_to_rotmatrix(euler: torch.Tensor) -> torch.Tensor:
    """
    Converts batch of euler rotations to batch of rotmatrix

    Args:
        euler (torch.Tensor): N*3 with roll, pitch, yaw, expressed in radian

    Returns:
        torch.Tensor: rotation representation with shapes N*3*3
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

    Args:
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

    Args:
        quat (torch.Tensor): rotation representation with shapes N*4

    Returns:
        torch.Tensor: rotation representation with shapes N*3*3
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
    """Converts a rotation matrix to quaternion
    torch implementation with logic from https://github.com/scipy/scipy/blob/7cb3d751756907238996502b92709dc45e1c6596/scipy/spatial/transform/rotation.py#L480
    WARNING: You cannot use this method in backpropagation as it uses inplace operations

    Args:
        rotmatrix (torch.Tensor): Rotmatrix with shape (N, 3, 3)

    Returns:
        torch.Tensor: Quaternion with shape (N, 4)
    """
    diag_elements = rotmatrix.diagonal(dim1=1, dim2=2)
    sum_diag = diag_elements.sum(dim=1, keepdim=True)
    decision_matrix = torch.cat([diag_elements, sum_diag], dim=1)
    indices = decision_matrix.argmax(dim=1)
    # init empty quat
    quat = torch.zeros((rotmatrix.shape[0], 4), device=rotmatrix.device)
    indice = torch.nonzero(indices != 3)
    i = indices[indice]
    j = (i + 1) % 3
    k = (j + 1) % 3
    quat[indice, i] = 1 - decision_matrix[indice, -1] + 2 * rotmatrix[indice, i, i]
    quat[indice, j] = rotmatrix[indice, j, i] + rotmatrix[indice, i, j]
    quat[indice, k] = rotmatrix[indice, k, i] + rotmatrix[indice, i, k]
    quat[indice, 3] = rotmatrix[indice, k, j] - rotmatrix[indice, j, k]
    indice = torch.nonzero(indices == 3)
    quat[indice, 0] = rotmatrix[indice, 2, 1] - rotmatrix[indice, 1, 2]
    quat[indice, 1] = rotmatrix[indice, 0, 2] - rotmatrix[indice, 2, 0]
    quat[indice, 2] = rotmatrix[indice, 1, 0] - rotmatrix[indice, 0, 1]
    quat[indice, 3] = 1 + decision_matrix[indice, -1]
    return normalize_quaternion(quat)


def normalize_quaternion(quat_t: torch.Tensor) -> torch.Tensor:
    """normalizes a quaternion tensor

    Args:
        quat_t (torch.Tensor): tensor quaternion

    Returns:
        torch.Tensor: normalized tensor quaternion
    """
    quat_normed = quat_t / quat_t.norm(dim=-1, keepdim=True)
    # Ensure we have the quaternion with a positive w to avoid double cover
    indices = torch.nonzero(quat_normed[..., -1] < 0, as_tuple=True)
    quat_normed[indices] = -quat_normed[indices]
    return quat_normed


def normalize_with_torch(t_tensor: torch.Tensor) -> torch.Tensor:
    """Normalize along the last dimension.

    Args:
        t_tensor (torch.Tensor): tensor to normalize

    Returns:
        torch.Tensor: normalized tensor
    """
    return t_tensor / t_tensor.norm(dim=-1, keepdim=True)


def rep6d_to_rotmatrix(rep6d: torch.Tensor) -> torch.Tensor:
    """
    Converts a batch of rep6d to rotmatrix

    Args:
        rep6d (torch.Tensor): rotation representation with shapes N*3*2

    Returns:
        torch.Tensor: rotation representation with shapes N*3*3
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

    Args:
        rep6d (torch.Tensor): rotation representation with shapes N*3*3

    Returns:
        torch.Tensor: rotation representation with shapes N*3*2
    """
    # remove last dimension on last axis
    rep6d = torch.narrow(rotmatrix, -1, 0, rotmatrix.shape[1] - 1)
    return rep6d


def _squeeze_batch(func):
    """decorator to squeeze a batch if any

    Args:
        func (_type_): function to decorate
    """

    def reshape_tensors(rotation_tensor: torch.Tensor):
        shape = rotation_tensor.shape
        if len(shape) > 2:
            rotation_tensor = rotation_tensor.reshape(-1, shape[-1])
        rotation_tensor = func(rotation_tensor)
        if len(shape) > 2:
            rotation_tensor = rotation_tensor.reshape(
                *shape[:-1], rotation_tensor.shape[-1]
            )
        return rotation_tensor

    return reshape_tensors


@_squeeze_batch
def convert_to_rotmatrix(rotation_tensor: torch.Tensor) -> torch.Tensor:
    """
    Converts a rotation tensor to rotmatrix

    Args:
        rotation_tensor(torch.Tensor): N * F with F in [3, 4, 6, 9]
    Returns:
        torch.Tensor: rotation tensor with shapes N * 9
    """
    if rotation_tensor.shape[-1] == ROTMATRIX_FEATURE_SIZE:
        return rotation_tensor
    if rotation_tensor.shape[-1] == REP6D_FEATURE_SIZE:
        return rep6d_to_rotmatrix(rotation_tensor.reshape(-1, 3, 2)).reshape(-1, 9)
    if rotation_tensor.shape[-1] == QUAT_FEATURE_SIZE:
        return quat_to_rotmatrix(rotation_tensor).reshape(-1, 9)
    if rotation_tensor.shape[-1] == EULER_FEATURE_SIZE:
        return euler_to_rotmatrix(rotation_tensor).reshape(-1, 9)


@_squeeze_batch
def convert_to_rep6d(rotation_tensor: torch.Tensor) -> torch.Tensor:
    """
    Converts a rotmatrix to rep6d

    Args:
        rotation_tensor(torch.Tensor): N * F with F in [3, 4, 6, 9]
    Returns:
        torch.Tensor: rotation tensor with shapes N * 6
    """
    if rotation_tensor.shape[-1] == REP6D_FEATURE_SIZE:
        return rotation_tensor
    if rotation_tensor.shape[-1] != ROTMATRIX_FEATURE_SIZE:
        rotation_tensor = convert_to_rotmatrix(rotation_tensor)
    rotation_tensor = rotation_tensor.reshape(-1, 3, 3)
    return rotmatrix_to_rep6d(rotation_tensor).reshape(-1, 6)


@_squeeze_batch
def convert_to_quat(rotation_tensor: torch.Tensor) -> torch.Tensor:
    """
    Converts a rotmatrix to quat

    Args:
        rotation_tensor(torch.Tensor): N * F with F in [3, 4, 6, 9]
    Returns:
        torch.Tensor: rotation tensor with shapes N * 4
    """
    if rotation_tensor.shape[-1] == QUAT_FEATURE_SIZE:
        return rotation_tensor
    if rotation_tensor.shape[-1] != ROTMATRIX_FEATURE_SIZE:
        rotation_tensor = convert_to_rotmatrix(rotation_tensor)
    rotation_tensor = rotation_tensor.reshape(-1, 3, 3)
    return rotmatrix_to_quat(rotation_tensor)


@_squeeze_batch
def convert_to_euler(rotation_tensor: torch.Tensor) -> torch.Tensor:
    """
    Converts a rotmatrix to euler

    Args:
        rotation_tensor(torch.Tensor): N * F with F in [3, 4, 6, 9]
    Returns:
        torch.Tensor: rotation tensor with shapes N * 3
    """
    if rotation_tensor.shape[-1] == EULER_FEATURE_SIZE:
        return rotation_tensor
    if rotation_tensor.shape[-1] != ROTMATRIX_FEATURE_SIZE:
        rotation_tensor = convert_to_rotmatrix(rotation_tensor)
    rotation_tensor = rotation_tensor.reshape(-1, 3, 3)
    return rotmatrix_to_euler(rotation_tensor)


def convert_rotation_tensor_to(
    tensor: torch.Tensor, rotation_rep: Rotation
) -> torch.Tensor:
    """Upper level function to map conversion method given rot_rep

    Args:
        tensor (torch.Tensor): the rotation tensor to convert
        rotation_rep (RotationRepresentation): format to convert tensor to

    Returns:
        torch.Tensor: new tensor
    """
    if rotation_rep is None:
        new_tensor = torch.zeros(*tensor.shape[:-1], 0)
    elif isinstance(rotation_rep, RotationEuler):
        new_tensor = convert_to_euler(tensor)
    elif isinstance(rotation_rep, RotationQuat):
        new_tensor = convert_to_quat(tensor)
        new_tensor = rotation_rep.post_process(new_tensor)
    elif isinstance(rotation_rep, RotationRotMat):
        new_tensor = convert_to_rotmatrix(tensor)
    elif isinstance(rotation_rep, RotationRep6D):
        new_tensor = convert_to_rep6d(tensor)
    else:
        raise AttributeError(f"{rotation_rep} is not an handled Rotation Feature")
    return new_tensor


def get_tensor_rotation_representation(tensor: torch.Tensor) -> Rotation:
    """Return the RotationRepresentation of a tensor based on its shapes

    Args:
        tensor (torch.Tensor): tensor with rotation only

    Returns:
        RotationRepresentation: the rotation representation of the input tensor, or None if tensor is empty
    """
    if tensor.shape[-1] == 0:
        return None
    if tensor.shape[-1] == ROTMATRIX_FEATURE_SIZE:
        return RotationRotMat
    if tensor.shape[-1] == REP6D_FEATURE_SIZE:
        return RotationRep6D
    if tensor.shape[-1] == QUAT_FEATURE_SIZE:
        return RotationQuat
    if tensor.shape[-1] == EULER_FEATURE_SIZE:
        return RotationEuler
    raise NotImplementedError(
        f"Found no matching representation for shape {tensor.shape[-1]}"
    )


def get_relative_rotation_from(
    input_tensor: torch.Tensor,
    basis_tensor: torch.Tensor,
    rotation_rep: Rotation = None,
) -> torch.Tensor:
    """return a new tensor relative to basis tensor using rotmatrices rotation

    Args:
        input_tensor (torch.Tensor): the input tensor to update (B, S, P, D)
        basis_tensor (torch.Tensor): the new basis tensor (B, [1:S], P, D)

    Returns:
        torch.Tensor: tensor relative to new basis
    """
    if not isinstance(rotation_rep, RotationRotMat):
        input_tensor = convert_rotation_tensor_to(
            input_tensor, RotationRotMat(range(9))
        )
        basis_tensor = convert_rotation_tensor_to(
            basis_tensor, RotationRotMat(range(9))
        )
    input_tensor = input_tensor.reshape(*input_tensor.shape[:-1], 3, 3)
    input_tensor = input_tensor.transpose(3, 4)
    basis_tensor = basis_tensor.reshape(*basis_tensor.shape[:-1], 3, 3).expand(
        input_tensor.shape
    )
    input_tensor = torch.matmul(input_tensor, basis_tensor)
    input_tensor = input_tensor.transpose(3, 4)
    input_tensor = input_tensor.reshape(*input_tensor.shape[:-2], 9)
    if not isinstance(rotation_rep, RotationRotMat):
        input_tensor = convert_rotation_tensor_to(input_tensor, rotation_rep)
    return input_tensor


def get_absolute_rotation_from(
    input_tensor: torch.Tensor,
    basis_tensor: torch.Tensor,
    rotation_rep: Rotation = None,
) -> torch.Tensor:
    """return a new tensor absolute to world based on basis tensor using rotmatrices rotation

    Args:
        input_tensor (torch.Tensor): the input tensor to update (N * D) with D representing a rotation
        basis_tensor (torch.Tensor): the new basis tensor (N * D) with D representing a rotation
    Returns:
        torch.Tensor: tensor absolute to world
    """
    if not isinstance(rotation_rep, RotationRotMat):
        input_tensor = convert_rotation_tensor_to(
            input_tensor, RotationRotMat(range(9))
        )
        basis_tensor = convert_rotation_tensor_to(
            basis_tensor, RotationRotMat(range(9))
        )
    input_tensor = input_tensor.reshape(*input_tensor.shape[:-1], 3, 3)
    basis_tensor = basis_tensor.reshape(*basis_tensor.shape[:-1], 3, 3).expand(
        input_tensor.shape
    )
    input_tensor[:, :, :] = torch.matmul(input_tensor[:, :, :], basis_tensor[:, :, :])
    input_tensor = input_tensor.reshape(*input_tensor.shape[:-2], 9)
    if not isinstance(rotation_rep, RotationRotMat):
        input_tensor = convert_rotation_tensor_to(input_tensor, rotation_rep)
    return input_tensor
