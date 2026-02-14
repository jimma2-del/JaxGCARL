from typing import Optional
from brax import actuator
from brax import com
from brax import contact
from brax import fluid
from brax import kinematics
from brax.base import Motion, System, Force
from brax.io import mjcf
from brax.spring import collisions
from brax.spring import integrator
from brax.spring import joints
from brax.spring.base import State

from brax.spring import pipeline as s_pipeline
import jax
import jax.numpy as jnp

original_pipeline_step = s_pipeline.step

def custom_pipeline_step(sys: System, state: State, act: jax.Array, debug: bool = False):
  print("CUSTOM STEP")
  #return original_pipeline_step(*args)

  tau = actuator.to_tau(sys, act, state.q, state.qd)
  xdd_i = Motion.create(vel=sys.gravity)
  xf_i = joints.resolve(sys, state, tau)
  if sys.enable_fluid:
    inertia = sys.link.inertia.i ** (1 - sys.spring_inertia_scale)
    xf_i += fluid.force(sys, state.x, state.xd, state.mass, inertia)

  # ADD CUSTOM FORCES (to xf_i)
  xf_i += Force.create(vel=jnp.zeros_like(xf_i.vel).at[0,2].add(100))

  xdd_i += Motion(
      ang=jax.vmap(lambda x, y: x @ y)(state.i_inv, xf_i.ang),
      vel=jax.vmap(lambda x, y: x / y)(xf_i.vel, state.mass),
  )

  # semi-implicit euler: apply acceleration update before resolving collisions
  state = state.replace(xd_i=state.xd_i + xdd_i * sys.opt.timestep)
  xdv_i = collisions.resolve(sys, state)

  # now integrate and update position/velocity-level terms
  x_i, xd_i = integrator.integrate(sys, state.x_i, state.xd_i, xdv_i)
  x, xd = com.to_world(sys, x_i, xd_i)
  state = state.replace(x=x, xd=xd, x_i=x_i, xd_i=xd_i)
  j, jd, a_p, a_c = kinematics.world_to_joint(sys, x, xd)
  q, qd = kinematics.inverse(sys, j, jd)
  state = state.replace(
      q=q,
      qd=qd,
      a_p=a_p,
      a_c=a_c,
      j=j,
      jd=jd,
      contact=contact.get(sys, x) if debug else None,
  )

  return state