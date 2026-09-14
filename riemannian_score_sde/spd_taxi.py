"""Taxi conditional score with a train-only empirical forward terminal.

Terminal draws are independent of predictors, matching the finance empirical
terminal approximation. This is not an exact conditional terminal distribution.
"""
import jax.numpy as jnp
from riemannian_score_sde.spd_sde import SPDBrownian


class TaxiSPDBrownian(SPDBrownian):
    def bind_training_data(self, batch):
        x, context = batch
        if context.shape != (len(x), 13):
            raise ValueError('Expected 13 Taxi predictors')
        self.limiting.data = jnp.asarray(x)
