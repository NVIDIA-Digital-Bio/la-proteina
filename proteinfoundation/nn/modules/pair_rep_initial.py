import torch

from proteinfoundation.nn.feature_factory import FeatureFactory
from proteinfoundation.nn.modules.adaptive_ln_scale import AdaptiveLayerNorm


class PairReprBuilder(torch.nn.Module):
    """
    Builds initial pair representation. Essentially the pair feature factory, but potentially with
    an adaptive layer norm layer as well.
    """

    def __init__(self, feats_repr, feats_cond, dim_feats_out, dim_cond_pair, **kwargs):
        super().__init__()

        self.init_repr_factory = FeatureFactory(
            feats=feats_repr,
            dim_feats_out=dim_feats_out,
            use_ln_out=True,
            mode="pair",
            **kwargs,
        )

        self.cond_factory = None  # Build a pair feature for conditioning and use it for adaln the pair representation
        if feats_cond is not None:
            if len(feats_cond) > 0:
                self.cond_factory = FeatureFactory(
                    feats=feats_cond,
                    dim_feats_out=dim_cond_pair,
                    use_ln_out=True,
                    mode="pair",
                    **kwargs,
                )
                self.adaln = AdaptiveLayerNorm(
                    dim=dim_feats_out, dim_cond=dim_cond_pair
                )

    def forward(self, batch_nn):
        mask = batch_nn["mask"]  # [b, n]
        pair_mask = mask[:, :, None] * mask[:, None, :]  # [b, n, n]
        repr = self.init_repr_factory(batch_nn)  # [b, n, n, dim_feats_out]
        if self.cond_factory is not None:
            cond = self.cond_factory(batch_nn)  # [b, n, n, dim_cond]
            repr = self.adaln(repr, cond, pair_mask)
        return repr

