import torch


class GeodesicLoss(torch.nn.modules.loss._Loss):
    """Compute the geodesic loss between two sets of rotation matrices."""

    def __init__(
        self, size_average=None, reduce=None, reduction: str = "mean", eps: float = 1e-7
    ) -> None:
        super(GeodesicLoss, self).__init__(size_average, reduce, reduction)
        self.eps = eps
        self.reduction = reduction

    def forward(
        self, rotmatrix_inputs: torch.Tensor, rotmatrix_targets: torch.Tensor
    ) -> torch.Tensor:
        """forward method computing the loss

        Args:
            rotmatrix_inputs (torch.Tensor): rotation_matrix to compare with truth
            rotmatrix_targets (torch.Tensor): truth rotation matrix

        Returns:
            torch.Tensor: geodesic distance bewteen tensors
        """
        # Compute the trace of the product of the two matrices
        # This computes the sum of the diagonal elements of the matrix product for each pair of matrices
        rotmatrix_inputs = torch.reshape(rotmatrix_inputs, [-1, 3, 3])
        rotmatrix_targets = torch.reshape(rotmatrix_targets, [-1, 3, 3])
        R_diffs = rotmatrix_inputs @ rotmatrix_targets.permute(0, 2, 1)
        all_traces = R_diffs.diagonal(dim1=-2, dim2=-1).sum(-1)
        # Clip the trace to ensure it is within the valid range for arcos
        all_traces = torch.clamp((all_traces - 1) / 2, -1.0 + self.eps, 1.0 - self.eps)
        # Compute the loss
        loss = torch.arccos(all_traces)
        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        return loss
