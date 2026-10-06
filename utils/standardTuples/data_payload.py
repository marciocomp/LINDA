# @author: Marcio Lopes
import torch

class ForwardDataTuple:
    """
    Payload for the Forward Pass (Data flowing from Tier 1 -> Tier 2 -> Tier 3).
    """
    def __init__(self,
                 tensor_data: torch.Tensor,
                 metadata: dict = None,
                 requires_grad: bool = True):
        """
        Args:
            tensor_data: The activation tensor (output of the previous split).
            metadata: Dictionary containing tracking info (e.g., 'source_id', 'batch_id', 'split_layer').
            requires_grad: Flag to indicate if this tensor needs gradients in the backward pass.
        """
        self.tensor_data = tensor_data
        self.metadata = metadata if metadata else {}
        self.requires_grad = requires_grad

    def to_cpu(self):
        """Moves tensor to CPU for serialization (Pickle compatibility)."""
        if isinstance(self.tensor_data, torch.Tensor):
            self.tensor_data = self.tensor_data.detach().cpu()
            # Note: We detach because we don't send the computational graph over the network.
            # We reconstruct the graph dependency on the receiver side.
        return self

class BackwardGradTuple:
    """
    Payload for the Backward Pass (Gradients flowing from Tier 3 -> Tier 2 -> Tier 1).
    """
    def __init__(self,
                 grad_tensor: torch.Tensor,
                 metadata: dict = None):
        """
        Args:
            grad_tensor: The gradient of the loss w.r.t the split layer output.
            metadata: Dictionary matching the ForwardDataTuple (to sync batch/source).
        """
        self.grad_tensor = grad_tensor
        self.metadata = metadata if metadata else {}

    def to_cpu(self):
        if isinstance(self.grad_tensor, torch.Tensor):
            self.grad_tensor = self.grad_tensor.detach().cpu()
        return self