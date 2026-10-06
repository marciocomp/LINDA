# @author: Marcio Lopes
# utils/standardTuples/split_matrix_line_tuple.py

class SplitMatrixLineTuple:
    """
    Represents a single row in the Global Split Matrix (Sigma).
    Corresponds to the allocation decision for a specific Data Source (Client).
    """
    def __init__(self,
                 tier1_split_point: int,
                 tier2_distribution: dict,
                 tier3_target: str):
        """
        Args:
            tier1_split_point (int): Index of the last layer executed on the Edge Device.
            tier2_distribution (dict): Allocation map for Tier 2 Pool {server_id: layer_count}.
            tier3_target (str): ID of the Tier 3 Node (Leader) for final aggregation or offloading.
        """
        self.tier1_split_point = tier1_split_point
        self.tier2_distribution = tier2_distribution
        self.tier3_target = tier3_target

    def to_dict(self):
        """Returns a dictionary representation for serialization/logging."""
        return {
            "tier1_split": self.tier1_split_point,
            "tier2_chain": self.tier2_distribution,
            "tier3_target": self.tier3_target
        }

    def __repr__(self):
        return str(self.to_dict())