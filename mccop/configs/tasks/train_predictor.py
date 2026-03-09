import math

from hydra_zen import builds, store

from mccop.configs import data, predictor, trainer  # noqa: F401
from mccop.configs.base import JobConfig, RunConfig, job_store
from mccop.train_predictor import main
from mccop.utils.runs import SweepJob

MainConfig = builds(main, cfg=RunConfig, populate_full_signature=True)

store(
    MainConfig,
    name="root",
    hydra_defaults=[
        "_self_",
        {"model": "mlp"},
        {"dataset": "tape_fluorescence"},
        {"trainer": "default"},
        {"cfg/wandb": None},
        {"cfg/job": None},
    ],
)

mlp_sweep = {
    "model.learning_rate": [1e-5],
    "model.hidden_dims": ["[512, 256]"],
    "model.dropout": [0.3],
}

cnn_sweep = {
    "model.learning_rate": [1e-5, 1e-4],
    "model.num_filters": [64, 128],
    "model.kernel_size": [3, 5],
}

resnet1d_sweep = {
    "model.learning_rate": [1e-5, 1e-4],
    "model.block_channels": ["[64, 128]", "[128, 256]"],
    "model.dropout": [0.2, 0.4],
}

resnet2d_sweep = {
    "model.learning_rate": [1e-5, 1e-4],
    "model.block_channels": ["[32, 64]"],
    "model.dropout": [0.2],
}

MODEL_SWEEPS = {
    "mlp": mlp_sweep,
    "cnn": cnn_sweep,
    "resnet1d": resnet1d_sweep,
    "resnet2d": resnet2d_sweep,
}

DATASETS = ["tape_fluorescence", "tape_stability", "ube4b"]
EMBED_DIM = 1024

for dataset_name in DATASETS:
    for model_name, sweep_params in MODEL_SWEEPS.items():
        parameters = {
            "cfg.seed": [42, 137, 67],
            "dataset": [dataset_name],
            "model": [model_name],
            **sweep_params,
        }
        num_combinations = math.prod(len(v) for v in parameters.values())

        project_name = f"{dataset_name}_{EMBED_DIM}_classification"
        sweep_config = builds(
            SweepJob,
            num_workers=num_combinations,
            parameters=parameters,
            project_name=project_name,
            builds_bases=(JobConfig,),
        )
        job_store(sweep_config, name=f"{dataset_name}_{model_name}")
