from hydra_zen import builds, store

from mccop.utils.runs import Job, Run, SlurmParams
from mccop.utils.wandb import WandBRun

RunConfig = builds(Run, seed=42, wandb=None, job=None)

SlurmParamsConfig = builds(
    SlurmParams,
    partition="gpu-5h",
    time_hours=5,
    cpus_per_task=8,
    gpus_per_task=1,
    mem_gb=360,
    nodes=1,
    tasks_per_node=1,
    constraint="80gb|h100",
)

SlurmParamsConfig2 = builds(
    SlurmParams,
    partition="gpu-2d",
    time_hours=48,
    cpus_per_task=8,
    gpus_per_task=1,
    mem_gb=128,
    nodes=1,
    tasks_per_node=1,
    constraint="80gb|h100",
)

JobConfig = builds(Job, slurm_params=SlurmParamsConfig)
job_store = store(group="cfg/job")
job_store(JobConfig, name="base")

WandBConfig = builds(WandBRun, group=None, mode="online")
wandb_store = store(group="cfg/wandb")
wandb_store(WandBConfig, name="base")
