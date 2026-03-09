from hydra_zen import builds, store

from mccop.configs.predictor import ClassificationMetrics
from mccop.smoothing.smoother import Smoother

smoother_store = store(group="smoother")

SmootherConfig = builds(
    Smoother,
    use_spectral_norm=True,
    use_jacobian_reg=True,
    use_distillation=False,
    use_smooth_activations=True,
    distillation_temp=2.0,
    distillation_alpha=0.5,
    reinitialize=True,
    learning_rate=1e-5,
    use_adversarial_aug=True,
    adv_epsilon=0.1,
    max_epochs=50,
    force_recompute=False,
    metrics=ClassificationMetrics,
    populate_full_signature=True,
)

smoother_store(
    SmootherConfig,
    name="base",
)
