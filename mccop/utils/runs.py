import subprocess
import sys
import tempfile
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

import yaml
from submitit import AutoExecutor
from submitit.helpers import CommandFunction

from mccop.utils.helpers import ConfigKeys, get_output_dir, logger
from mccop.utils.wandb import WandBConfig, WandBRun


@dataclass
class SlurmParams:
    """Slurm resource configuration."""

    partition: str | None = None
    cpus_per_task: int | None = None
    gpus_per_task: int | None = None
    mem_gb: int | None = None
    excluded_nodes: list[str] = field(default_factory=list)
    constraint: str | None = None
    time_hours: int | None = None
    nodes: int | None = None
    tasks_per_node: int | None = None
    tmp: str | None = None

    def to_submitit_params(self) -> dict[str, Any]:
        """Convert to submitit parameters."""
        params: dict[str, Any] = {}
        for param in fields(self):
            if (value := getattr(self, param.name)) is not None:
                match param.name:
                    case "excluded_nodes":
                        params["slurm_exclude"] = ",".join(value)
                    case "time_hours":
                        params["slurm_time"] = f"{value}:00:00"
                    case _:
                        params[f"slurm_{param.name}"] = value
        return params


@dataclass
class Job:
    """Job to run code on a cluster using apptainer."""

    image: str = "oras://ghcr.io/weroks/mccop:latest-sif"
    data_dir: str = "datasets"
    cluster: str = "slurm"
    slurm_params: SlurmParams = field(default_factory=SlurmParams)
    wait_for_job: bool = False
    timeout_min: int = 5
    project_name: str | None = None

    def __post_init__(self) -> None:
        """Run the job."""
        self.run()
        sys.exit(0)

    def filter_args(self, args: list[str]) -> list[str]:
        """Filter args to prevent recursive jobs on the cluster."""
        return [arg for arg in args if f"{ConfigKeys.CONFIG}/{ConfigKeys.JOB}" not in arg]

    @property
    def project(self) -> str:
        """Get the project name from the job or the environment."""
        if self.project_name:
            return self.project_name
        if (wandb_config := WandBConfig.from_env()) and wandb_config.WANDB_PROJECT:
            return wandb_config.WANDB_PROJECT
        return "runs"

    @property
    def python_command(self) -> str:
        """Python command used by the job."""
        base_command = "apptainer run --nv"
        if Path(self.data_dir).exists():
            logger.info(f"Binding data directory {self.data_dir} to container.")
            return f"{base_command} -B {self.data_dir} {self.image} python"
        return f"{base_command} {self.image} python"

    def run(self) -> None:
        """Run the job on the cluster."""
        output_dir = get_output_dir()

        command = [
            "python",
            *self.filter_args(sys.argv),
            "cfg/wandb=base",
            f"hydra.run.dir={output_dir}",
        ]

        function = CommandFunction(command)
        executor = AutoExecutor(
            folder=output_dir,
            cluster=self.cluster,
            slurm_python=self.python_command,
        )

        executor.update_parameters(timeout_min=self.timeout_min, **self.slurm_params.to_submitit_params())
        job = executor.submit(function)

        logger.info(f"Submitted job {job.job_id} to folder {output_dir}")

        if self.wait_for_job:
            logger.info(f"\n{job.result()}")


@dataclass
class SweepJob(Job):
    """Job to run a sweep on a cluster."""

    num_workers: int = 1
    parameters: dict[str, list[Any]] = field(default_factory=dict)
    metric_name: str = "loss"
    metric_goal: Literal["maximize", "minimize"] = "minimize"

    @property
    def project(self) -> str:
        """Get the project name from the job or the environment."""
        if self.project_name:
            return self.project_name
        if (wandb_config := WandBConfig.from_env()) and wandb_config.WANDB_PROJECT:
            return wandb_config.WANDB_PROJECT
        raise RuntimeError("No project name found in job or WandB config.")

    def register_sweep(self, sweep_config: dict) -> str:
        """Register a wandb sweep from a config."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "sweep_config.yaml"

            with Path.open(config_path, "w") as config_file:
                yaml.dump(sweep_config, config_file)

            output = subprocess.run(
                ["wandb", "sweep", "--project", self.project, str(config_path)],
                check=True,
                text=True,
                capture_output=True,
            ).stderr

            sweep_id = output.split(" ")[-1].strip()

            for line in output.splitlines():
                logger.info(line)

        return sweep_id

    def run(self) -> None:
        """Run the sweep on the cluster."""
        parameters = {cfg_key: {"values": list(values)} for cfg_key, values in self.parameters.items()}
        metric = {"goal": self.metric_goal, "name": self.metric_name}
        program, args = sys.argv[0], self.filter_args(sys.argv[1:])

        sweep_folder = get_output_dir()

        command = [
            "${env}",
            "${interpreter}",
            "${program}",
            *args,
            "cfg/wandb=base",
            "hydra.run.dir=" + str(sweep_folder / "${now:%M-%S-%f}"),
            "${args_no_hyphens}",
        ]

        sweep_config = {
            "program": program,
            "method": "grid",
            "metric": metric,
            "parameters": parameters,
            "command": command,
        }

        sweep_id = self.register_sweep(sweep_config)

        function = CommandFunction(["wandb", "agent", "--count", "1"])
        executor = AutoExecutor(
            folder=sweep_folder,
            cluster=self.cluster,
            slurm_python=self.python_command,
        )
        executor.update_parameters(
            slurm_array_parallelism=self.num_workers,
            **self.slurm_params.to_submitit_params(),
        )

        jobs = executor.map_array(function, [sweep_id] * self.num_workers)

        for job in jobs:
            logger.info(f"Submitted job {job.job_id} to sweep folder {sweep_folder}")


@dataclass
class Run:
    """Configures a basic run."""

    seed: int
    wandb: WandBRun | None = None
    job: Any | None = None
