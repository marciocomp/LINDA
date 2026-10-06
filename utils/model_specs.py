# @author: Marcio Lopes

def get_model_specs(model_name):
    """
    Returns estimated costs per split point for the model.
    Structure: List of dicts where index is the split layer.

    Estimates for ResNet50 (Cumulative):
    - FLOPS in GFLOPs (approx)
    - Memory in GB (Activations + Weights)
    - Output Size in MB (Smashed Data to transmit)
    """
    if model_name == "resnet50":
        # ResNet50Split has 6 blocks (Stem + Layer1..4 + Head)
        # These are heuristic values. Ideally, use a profiler pass.
        return [
            # Index 0: No local processing (Cloud-only) - Not allowed in SL usually
            {"flops": 0.0, "mem": 10.0, "out_mb": 0.15},

            # Index 1: Stem (Conv1)
            {"flops": 0.5, "mem": 100.2, "out_mb": 3.0},

            # Index 2: Layer 1
            {"flops": 1.5, "mem": 1000.5, "out_mb": 3.0},

            # Index 3: Layer 2
            {"flops": 3.0, "mem": 1000.2, "out_mb": 1.5},

            # Index 4: Layer 3
            {"flops": 5.0, "mem": 1002.5, "out_mb": 0.8},

            # Index 5: Layer 4
            {"flops": 7.5, "mem": 1003.5, "out_mb": 0.1},

            # Index 6: Full Model (Local Training)
            {"flops": 8.0, "mem": 1004.0, "out_mb": 0.001}
        ]
    elif model_name == "gemma":
        return []
    else:
        # Default fallback
        return [{"flops": 1.0, "mem": 1.0, "out_mb": 1.0}] * 5