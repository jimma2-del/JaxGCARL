from brax.spring import base

from flax import struct
import jax

@struct.dataclass
class State(base.State):
  """Custom state with antagonist's action added

  Attributes:
    antag_action: (antagonist num_actions, )
  """
  antag_action: jax.Array