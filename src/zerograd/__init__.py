"""JAX + Optax primitives for zero-gradient evolutionary optimization."""

from ._candidate import CandidateContext, perturbed_linear, perturbed_table_lookup, perturbed_tied_logits, perturbed_vector
from ._cluster import ClusterZeroGrad, ZeroGradNode
from ._distributed import CalibrationResult, DeviceShard, DistributedZeroGrad, ShardResult, compute_partition_sizes
from ._fault_tolerant import FaultTolerantCluster, NodeStatus
from ._factors import matrix_factors, scaled_factor, table_factors, vector_noise
from ._fitness import shape_centered_loss, validate_losses
from ._keys import candidate_key, group_key, step_key
from ._manifest import Manifest, ManifestEntry, ParameterLayout, ParameterPath, ParameterTree
from ._optimizer import StepMetrics, ZeroGrad, ZeroGradState
from ._replay import replay, replay_entry

__all__ = [
    "CandidateContext",
    "CalibrationResult",
    "ClusterZeroGrad",
    "DeviceShard",
    "DistributedZeroGrad",
    "FaultTolerantCluster",
    "Manifest",
    "ManifestEntry",
    "NodeStatus",
    "ParameterLayout",
    "ParameterPath",
    "ParameterTree",
    "ShardResult",
    "StepMetrics",
    "ZeroGrad",
    "ZeroGradNode",
    "ZeroGradState",
    "candidate_key",
    "compute_partition_sizes",
    "group_key",
    "matrix_factors",
    "perturbed_linear",
    "perturbed_table_lookup",
    "perturbed_tied_logits",
    "perturbed_vector",
    "replay",
    "replay_entry",
    "scaled_factor",
    "shape_centered_loss",
    "step_key",
    "table_factors",
    "validate_losses",
    "vector_noise",
]
