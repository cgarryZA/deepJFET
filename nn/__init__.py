"""Neural-network surrogate pipeline for JFET gate optimization.

Public API
----------
    from nn import SurrogatePipeline, SamplingConfig, TrainConfig

    pipe = SurrogatePipeline(board_config)
    pipe.run(GateType.NAND2)          # generate data → train → optimize → verify
    design = pipe.designs[GateType.NAND2]
"""

from .config import SamplingConfig, TrainConfig, PipelineConfig
from .data import generate_dataset, load_dataset, save_dataset
from .model import GateSurrogateNet
from .train import train_surrogate, load_surrogate
from .surrogate import SurrogateOptimizer
from .pipeline import SurrogateGatePipeline
