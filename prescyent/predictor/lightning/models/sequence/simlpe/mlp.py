import torch
from torch import nn
from prescyent.utils.enums.trajectory_dimensions import TrajectoryDimensions


class CustomLayerNorm(nn.Module):
    """
    Custom Layer Normalization class.
    This class applies layer normalization across a specified axis of the input tensor.
    The normalization is parameterized with learnable scaling (alpha) and shifting (beta) parameters.

    Args:
        axis (int): The axis along which to apply normalization (e.g., 1 for spatial, -1 for temporal).
        size (int): The size of the axis to normalize.
        epsilon (float): A small value added to the variance for numerical stability.
    """

    def __init__(self, axis, size, epsilon=1e-5):
        super().__init__()
        self.axis = axis
        self.epsilon = epsilon
        _shape = [1, 1, 1]
        _shape[axis] = size
        self.alpha = nn.Parameter(torch.ones(_shape), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(_shape), requires_grad=True)

    def forward(self, x):
        """
        Forward pass for layer normalization.

        Args:
            x (torch.Tensor): Input tensor to normalize.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        mean = x.mean(axis=self.axis, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=self.axis, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y


class SpatialFC(nn.Module):
    """
    Fully connected layer that operates on the spatial dimension (transpose is applied).

    Args:
        dim (int): Input and output dimension size.
    """

    def __init__(self, dim):
        super(SpatialFC, self).__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        """
        Forward pass for spatial fully connected layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq, spatial_dim).

        Returns:
            torch.Tensor: Output tensor with fully connected applied.
        """
        x = torch.transpose(x, 1, 2)
        x = self.fc(x)
        x = torch.transpose(x, 1, 2)
        return x


class TemporalFC(nn.Module):
    """
    Fully connected layer that operates on the temporal dimension.

    Args:
        dim (int): Input and output dimension size.
    """

    def __init__(self, dim):
        super(TemporalFC, self).__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        """
        Forward pass for temporal fully connected layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq, dim).

        Returns:
            torch.Tensor: Output tensor with fully connected applied.
        """
        x = self.fc(x)
        return x


class MLPBlock(nn.Module):
    """
    A block consisting of either a spatial or temporal fully connected layer followed by normalization.

    The block supports different normalization strategies: spatial, temporal, batch, or layer normalization.

    Args:
        config: Configuration object that includes layer dimension sizes and normalization settings.
    """

    def __init__(self, config):
        super().__init__()

        # Initialize either spatial or temporal fully connected based on config.
        if config.spatial_fc_only:
            self.fc0 = SpatialFC(config.in_points_dims)
        else:
            self.fc0 = TemporalFC(config.in_sequence_size)
        if config.mpl_blocks_norm:
            if config.mpl_blocks_norm == TrajectoryDimensions.SPATIAL:
                self.norm0 = CustomLayerNorm(1, config.in_points_dims)
            elif config.mpl_blocks_norm == TrajectoryDimensions.TEMPORAL:
                self.norm0 = CustomLayerNorm(-1, config.in_sequence_size)
            elif config.mpl_blocks_norm == TrajectoryDimensions.ALL:
                self.norm0 = nn.LayerNorm(
                    [config.in_points_dims, config.in_sequence_size]
                )
            elif config.mpl_blocks_norm == TrajectoryDimensions.BATCH:
                self.norm0 = nn.BatchNorm1d(config.in_points_dims)
            else:
                raise NotImplementedError(
                    f"{config.mpl_blocks_norm} is not a valid norm type"
                )
        else:
            self.norm0 = nn.Identity()  # No normalization applied if not specified.

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initializes the fully connected layer's weights using Xavier uniform initialization
        with a very small gain (1e-8), and sets biases to zero.
        """
        nn.init.xavier_uniform_(self.fc0.fc.weight, gain=1e-8)
        nn.init.constant_(self.fc0.fc.bias, 0)

    def forward(self, x):
        """
        Forward pass for the MLP block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after the fully connected and normalization.
        """
        x_ = self.fc0(x)
        x_ = self.norm0(x_)
        x = x + x_
        return x


class TransMLP(nn.Module):
    """
    A multi-layer perceptron (MLP) model consisting of stacked MLPBlocks.

    Args:
        config: Configuration object with number of layers and dimensions.
    """

    def __init__(self, config):
        super().__init__()
        self.mlps = nn.Sequential(*[MLPBlock(config) for i in range(config.num_layers)])

    def forward(self, x):
        """
        Forward pass through the entire MLP model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after passing through all MLP blocks.
        """
        x = self.mlps(x)
        return x
