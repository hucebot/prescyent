import torch


class ReshapingLayer(torch.nn.Module):
    """creates a new linear layer to match new input shape to old input shapes"""

    def __init__(self, input_shapes, output_shapes) -> None:
        super().__init__()
        self.input_shapes = input_shapes
        self.output_shapes = output_shapes
        self.kept_dimensions = [
            i
            for i, i_shape in enumerate(self.input_shapes)
            if self.output_shapes[i] == i_shape
        ]
        if len(self.kept_dimensions) == len(self.input_shapes):
            return
        self.in_features = 1
        self.out_features = 1
        for dim in [
            i for i in range(len(self.input_shapes)) if i not in self.kept_dimensions
        ]:
            self.in_features = self.in_features * self.input_shapes[dim]
            self.out_features = self.out_features * output_shapes[dim]
        self.linear = torch.nn.Linear(self.in_features, self.out_features)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if len(self.kept_dimensions) == len(self.input_shapes):
            return input_tensor
        # flatten on the dimensions which's shapes has to be updated
        for i, dim in enumerate(self.kept_dimensions):
            input_tensor = torch.transpose(input_tensor, dim, 0 + i)
        input_tensor = torch.flatten(input_tensor, len(self.kept_dimensions))
        # update tensor values with linear layer
        output_tensor = self.linear(input_tensor)
        # unflatten on the dimensions which's shapes has to be updated
        return output_tensor.reshape(self.output_shapes)
