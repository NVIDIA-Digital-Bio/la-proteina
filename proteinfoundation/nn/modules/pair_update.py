import torch
from torch.utils.checkpoint import checkpoint

from openfold.model.pair_transition import PairTransition
from openfold.model.triangular_multiplicative_update import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)


class PairReprUpdate(torch.nn.Module):
    """Layer to update the pair representation."""

    def __init__(
        self,
        token_dim,
        pair_dim,
        expansion_factor_transition=2,
        use_tri_mult=False,
        tri_mult_c=196,
    ):
        super().__init__()

        self.use_tri_mult = use_tri_mult
        self.layer_norm_in = torch.nn.LayerNorm(token_dim)
        self.linear_x = torch.nn.Linear(token_dim, int(2 * pair_dim), bias=False)

        if use_tri_mult:
            tri_mult_c = min(pair_dim, tri_mult_c)
            self.tri_mult_out = TriangleMultiplicationOutgoing(
                c_z=pair_dim, c_hidden=tri_mult_c
            )
            self.tri_mult_in = TriangleMultiplicationIncoming(
                c_z=pair_dim, c_hidden=tri_mult_c
            )
        self.transition_out = PairTransition(
            c_z=pair_dim, n=expansion_factor_transition
        )

    def _apply_mask(self, pair_rep, pair_mask):
        """
        pair_rep has shape [b, n, n, pair_dim]
        pair_mask has shape [b, n, n]
        """
        return pair_rep * pair_mask[..., None]

    def forward(self, x, pair_rep, mask):
        """
        Args:
            x: Input sequence, shape [b, n, token_dim]
            pair_rep: Input pair representation, shape [b, n, n, pair_dim]
            mask: binary mask, shape [b, n]

        Returns:
            Updated pair representation, shape [b, n, n, pair_dim].
        """
        pair_mask = mask[:, None, :] * mask[:, :, None]  # [b, n, n]
        x = x * mask[..., None]  # [b, n, token_dim]
        x_proj_1, x_proj_2 = self.linear_x(self.layer_norm_in(x)).chunk(
            2, dim=-1
        )  # [b, n, pair_dim] each
        pair_rep = (
            pair_rep + x_proj_1[:, None, :, :] + x_proj_2[:, :, None, :]
        )  # [b, n, n, pair_dim]
        pair_rep = self._apply_mask(pair_rep, pair_mask)  # [b, n, n, pair_dim]
        if self.use_tri_mult:
            pair_rep = pair_rep + checkpoint(
                self.tri_mult_out, *(pair_rep, pair_mask * 1.0)
            )
            pair_rep = self._apply_mask(pair_rep, pair_mask)  # [b, n, n, pair_dim]
            pair_rep = pair_rep + checkpoint(
                self.tri_mult_in, *(pair_rep, pair_mask * 1.0)
            )
            pair_rep = self._apply_mask(pair_rep, pair_mask)  # [b, n, n, pair_dim]
        pair_rep = pair_rep + checkpoint(
            self.transition_out, *(pair_rep, pair_mask * 1.0)
        )
        pair_rep = self._apply_mask(pair_rep, pair_mask)  # [b, n, n, pair_dim]
        return pair_rep
