# @author: Marcio Lopes

class ResourceVectorTuple:
    """
    Represents the computational capabilities of a node (Algorithm 1 Output).
    """
    def __init__(self, flops: float, ram: float):
        """
        Args:
            flops: Floating Point Operations per Second (usually in TFLOPS).
            ram: Available Memory (in GB).
        """
        self.flops = flops
        self.ram = ram

    def to_dict(self):
        """Returns a string representation for logging."""
        return f"(FLOPS: {self.flops} , RAM: {self.ram})"

    def __repr__(self):
        return str(self.to_dict())