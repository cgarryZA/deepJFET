"""Neural-network surrogate pipeline for JFET gate optimization.

Public API
----------
    from nn import SurrogateGatePipeline, PipelineConfig, SamplingConfig, TrainConfig

    pipe = SurrogateGatePipeline(board_config)
    pipe.run(GateType.NAND2)          # generate data -> train -> optimize -> verify
    design = pipe.designs[GateType.NAND2]
"""

from .config import SamplingConfig, TrainConfig, PipelineConfig
from .data import generate_dataset, load_dataset, save_dataset
from .model import GateSurrogateNet, COLUMNS_X, COLUMNS_Y
from .train import train_surrogate, load_surrogate
from .surrogate import SurrogateOptimizer
from .pipeline import SurrogateGatePipeline
from .registry import find_model, register_model, jfet_hash
from .gpu_solver import gpu_solve_batch
from .gpu_transient import generate_delay_dataset, save_delay_dataset, load_delay_dataset
from .delay_model import train_delay_model, predict_delay
