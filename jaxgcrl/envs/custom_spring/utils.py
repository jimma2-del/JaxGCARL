import jax
from brax.base import Force, State
from jax import numpy as jnp

def make_ft(
    pipeline_state: State,
    body_id: int,
    point: jax.Array,
    force: jax.Array,
    torque: jax.Array,
) -> Force:
    """Approximate mjx.apply_ft for the spring backend at an arbitrary body point."""
    
    x_i = pipeline_state.x_i
    
    # COM position in world frame for this body
    com_pos = x_i.pos[body_id]
    # lever arm from COM to application point
    r = point - com_pos
    # torque about COM: external torque plus r x F
    tau_com = torque + jnp.cross(r, force)

    return Force(
        vel=jnp.zeros_like(x_i.pos).at[body_id].set(force), 
        ang=jnp.zeros_like(x_i.pos).at[body_id].set(tau_com)
    )