"""Class-conditional empirical terminal; financial experiments stay unchanged."""
import numpy as np
import jax.numpy as jnp
import jax.random as random
from riemannian_score_sde.spd_sde import SPDBrownian, EmpiricalForwardTerminal


class ConditionalTerminal(EmpiricalForwardTerminal):
    def sample(self, rng, shape):
        raise ValueError('EEG generation requires a class context')

    def sample_conditioned(self, rng, shape, context):
        if context is None:
            raise ValueError('Missing EEG class context')
        # Padded class index lists allow a single compiled graph for either label.
        labels = jnp.argmax(context,axis=-1)
        ik, fk = random.split(rng)
        offsets = jnp.floor(random.uniform(ik,(shape[0],))*self.counts[labels]).astype(jnp.int32)
        index = self.class_indices[labels, offsets]
        return self.sde.marginal_sample(fk,self.data[index],jnp.full((shape[0],),self.sde.tf))


class ConditionalSPDBrownian(SPDBrownian):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.limiting = ConditionalTerminal(self)

    def bind_training_data(self, batch):
        x, context = batch
        labels = np.argmax(np.asarray(context),axis=-1)
        lists = [np.flatnonzero(labels==i) for i in range(2)]
        if any(len(a)==0 for a in lists):
            raise ValueError('Both EEG classes required in training')
        self.limiting.data = jnp.asarray(x)
        self.limiting.counts = jnp.array([len(a) for a in lists])
        width=max(map(len,lists))
        self.limiting.class_indices=jnp.array(np.stack([np.pad(a,(0,width-len(a)),mode='edge') for a in lists]))
