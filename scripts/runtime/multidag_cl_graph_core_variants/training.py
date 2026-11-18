"""Only checkpoint identity differs; optimizer/loader/loss/selection are inherited."""

from copy import deepcopy

from models.multidag_cl.graph_core_variants import GraphCoreTA, VARIANTS
from scripts.runtime.multidag_cl_ta_v0.training import TrainDevRuntime


class GraphCoreRuntime(TrainDevRuntime):
    def __init__(self, model, *args, **kwargs):
        if not isinstance(model, GraphCoreTA) or not model.graph_config.enabled:
            raise ValueError("variant runtime requires one enabled independent GraphCoreTA")
        super().__init__(model, *args, **kwargs)

    @property
    def identity(self):
        result = super().identity
        result.update(schema="multidag_graph_core_train_dev_checkpoint_v1",
                      graph_core=deepcopy(self.model.graph_config.resolved()),
                      **VARIANTS[self.model.graph_config.variant])
        return result
