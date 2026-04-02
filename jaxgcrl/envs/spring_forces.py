import jax
from brax import base
from jax import numpy as jnp


def apply_ft_spring(
    pipeline_state: base.State,
    body_id: int,
    point: jax.Array,
    force: jax.Array,
    torque: jax.Array,
    dt: float,
) -> base.State:
    """Approximate mjx.apply_ft for the spring backend at an arbitrary body point.

    This operates in the spring pipeline's body-COM frame: it converts a world-frame
    force/torque applied at `point` on `body_id` into an equivalent wrench at the
    body's COM, then applies the resulting angular/linear impulse to xd_i for one
    timestep. The actual mapping back to joint space happens on the next pipeline step.
    """
    # spring.State exposes x_i (COM transforms), xd_i (COM motions),
    # per-link masses and inverse inertia tensors.
    x_i = pipeline_state.x_i
    xd_i = pipeline_state.xd_i
    mass = pipeline_state.mass
    i_inv = pipeline_state.i_inv

    # COM position in world frame for this body
    com_pos = x_i.pos[body_id]
    # lever arm from COM to application point
    r = point - com_pos
    # torque about COM: external torque plus r x F
    tau_com = torque + jnp.cross(r, force)

    # angular and linear accelerations at COM due to this wrench
    alpha = i_inv[body_id] @ tau_com
    a = force / mass[body_id]

    xd_i = xd_i.replace(
        ang=xd_i.ang.at[body_id].add(alpha * dt),
        vel=xd_i.vel.at[body_id].add(a * dt),
    )

    return pipeline_state.replace(xd_i=xd_i)

