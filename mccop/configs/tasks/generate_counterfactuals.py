import math

from hydra_zen import builds, store

from mccop.configs import baselines, data, mccop, smoother  # noqa: F401
from mccop.configs.base import JobConfig, RunConfig, job_store
from mccop.generate_counterfactuals import main
from mccop.utils.runs import SweepJob

MainConfig = builds(main, cfg=RunConfig, populate_full_signature=True)

store(
    MainConfig,
    name="root",
    hydra_defaults=[
        "_self_",
        {"editor": "mccop"},
        {"dataset": "tape_fluorescence"},
        {"cfg/wandb": None},
        {"cfg/job": None},
    ],
)

editor_sweep = {
    "editor.project_on_manifold": [True],
    "editor.sampling_time_fraction": [0.1],
    "editor.learning_rate": [5e-1, 1e-1, 5e-2],
    "editor/loss_fn": ["margin_loss", "bce_loss"],
    "editor/sparsity_mechanism": [
        "gradient_masking",
        "composite_masking_l2",
    ],
}

smoother_sweep = {
    "editor.smoother.use_spectral_norm": [True],
    "editor.smoother.use_adversarial_aug": [True],
    "editor.smoother.use_jacobian_reg": [True],
    "editor.smoother.use_distillation": [False],
    "editor.smoother.use_smooth_activations": [True],
    "editor.smoother.force_recompute": [True],
    "dataset": ["tape_fluorescence", "tape_stability", "ube4b"],
    "cfg.seed": [42, 43, 44],
}

smoother_baseline_sweep = {
    "editor.smoother.use_spectral_norm": [False],
    "editor.smoother.use_adversarial_aug": [False],
    "editor.smoother.use_jacobian_reg": [False],
    "editor.smoother.use_distillation": [False],
    "editor.smoother.use_smooth_activations": [False],
    "editor.smoother.force_recompute": [True],
    "dataset": ["tape_fluorescence", "tape_stability", "ube4b"],
    "cfg.seed": [42, 43, 44],
}

baseline_sweep = {
    "editor": ["baseline1", "baseline2", "baseline3"],
    "dataset": ["tape_fluorescence", "tape_stability", "ube4b"],
    "cfg.seed": [42, 43, 44],
}

test_sweep = {
    "cfg.seed": [42, 43, 44],
    "dataset": ["tape_fluorescence", "tape_stability", "ube4b"],
}

project_name = "baselines"
sweep_config = builds(
    SweepJob,
    num_workers=math.prod(len(v) for v in baseline_sweep.values()),
    parameters=baseline_sweep,
    project_name=project_name,
    builds_bases=(JobConfig,),
)
job_store(sweep_config, name="baselines")

project_name = "test_run"
sweep_config = builds(
    SweepJob,
    num_workers=math.prod(len(v) for v in test_sweep.values()),
    parameters=test_sweep,
    project_name=project_name,
    builds_bases=(JobConfig,),
)
job_store(sweep_config, name="test_run")

smoothing_eval_config = builds(
    SweepJob,
    num_workers=math.prod(len(v) for v in smoother_sweep.values()),
    parameters=smoother_sweep,
    project_name="smoother_evaluation",
    builds_bases=(JobConfig,),
)
job_store(smoothing_eval_config, name="smoother_evaluation")
