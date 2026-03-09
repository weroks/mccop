from dima import get_stats_path
from hydra_zen import builds, store
from torch import nn

from mccop.configs.smoother import SmootherConfig
from mccop.counterfactual.mccop import MCCOP
from mccop.counterfactual.losses import KLSparsityLoss, MarginLoss
from mccop.counterfactual.sparsity import (
    CompositeSparsity,
    DistanceSparsity,
    GradientMasking,
    KLSparsity,
)

# Group stores
editor_store = store(group="editor")
sparsity_store = store(group="editor/sparsity_mechanism")
loss_store = store(group="editor/loss_fn")

STATS_DIR = get_stats_path()

# Loss functions
bce_loss = builds(nn.BCEWithLogitsLoss)
margin_loss = builds(MarginLoss, margin=2.2)
loss_store(margin_loss, name="margin_loss")
loss_store(bce_loss, name="bce_loss")

# Sparsity losses/mechanisms
gradient_masking = builds(
    GradientMasking,
    k=5,
)
sparsity_store(gradient_masking, name="gradient_masking")

kl_inner_loss = builds(
    KLSparsityLoss,
    target_edits=2,
    temperature=5.0,
    penalty_too_many=100.0,
    penalty_too_few=10.0,
    sharpness=2.0,
)
kl_sparsity = builds(
    KLSparsity,
    loss_fn=kl_inner_loss,
    weight=1.0,
)
sparsity_store(kl_sparsity, name="kl_sparsity")

l2_inner_loss = builds(nn.MSELoss)
l2_sparsity = builds(
    DistanceSparsity,
    loss_fn=l2_inner_loss,
    weight=0.1,
)
sparsity_store(l2_sparsity, name="l2_sparsity")

composite_masking_l2 = builds(
    CompositeSparsity,
    mechanisms=[gradient_masking, l2_sparsity],
)
sparsity_store(composite_masking_l2, name="composite_masking_l2")

# Main experiment config
MCCOPConfig = builds(
    MCCOP,
    predictor=None,
    smoother=SmootherConfig,
    project_on_manifold=True,
    sampling_time_fraction=0.1,
    learning_rate=5e-1,
    gradient_steps=50,
    seed=42,
    original_class_filter=0,
    normalization_stats_path=STATS_DIR,
    populate_full_signature=True,
    zen_partial=True,
    hydra_defaults=[
        "_self_",
        {"loss_fn": "margin_loss"},
        {"sparsity_mechanism": "composite_masking_l2"},
    ]
)
editor_store(
    MCCOPConfig,
    name="mccop",
)
