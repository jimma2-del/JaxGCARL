# Copy of the brax.spring.pipeline, 
# with modifications to allow for custom forces, masses, and friction

from typing import Optional
from brax import actuator
from brax import com
from brax import contact
from brax import fluid
from brax import kinematics
from brax.base import Motion, System
from brax.io import mjcf
from brax.spring import collisions
from brax.spring import integrator
from brax.spring import joints

## <mark use our custom State instead of default spring State>
#from brax.spring.base import State
from .base import State
## </mark>

import jax

## <mark custom imports>
import jax.numpy as jnp

from .utils import make_ft
## </mark>

def init(
    sys: System,
    q: jax.Array,
    qd: jax.Array,
    unused_act: Optional[jax.Array] = None,
    unused_ctrl: Optional[jax.Array] = None,
    debug: bool = False,
) -> State:
    """Initializes physics state.
    
    Args:
    sys: a brax system
    q: (q_size,) joint angle vector
    qd: (qd_size,) joint velocity vector
    debug: if True, adds contact to the state for debugging
    
    Returns:
    state: initial physics state
    """
    if sys.mj_model is not None:
        mjcf.validate_model(sys.mj_model)
    # position/velocity level terms
    x, xd = kinematics.forward(sys, q, qd)
    j, jd, a_p, a_c = kinematics.world_to_joint(sys, x, xd)
    x_i, xd_i = com.from_world(sys, x, xd)
    i_inv = com.inv_inertia(sys, x)
    mass = sys.link.inertia.mass ** (1 - sys.spring_mass_scale)

    ## <mark change mass>
    #mass = mass.at[0].set(mass[0] * 100)
    #mass = mass.at[0].set(mass[0] * 0.1)
    mass = mass * 0.5
    ## </mark>
    
    return State(
        q=q,
        qd=qd,
        x=x,
        xd=xd,
        contact=contact.get(sys, x) if debug else None,
        x_i=x_i,
        xd_i=xd_i,
        j=j,
        jd=jd,
        a_p=a_p,
        a_c=a_c,
        i_inv=i_inv,
        mass=mass,

        ## <mark add antag_action>
        antag_action = jnp.zeros((3)),
        ## </mark>
    )


def step(
    sys: System, state: State, act: jax.Array, debug: bool = False
) -> State:
    """Performs a single physics step using spring-based dynamics.
    
    Resolves actuator forces, joints, and forces at acceleration level, and
    resolves collisions at velocity level with baumgarte stabilization.
    
    Args:
    sys: system defining the kinematic tree and other properties
    state: physics state prior to step
    act: (act_size,) actuator input vector
    debug: if True, adds contact to the state for debugging
    
    Returns:
    x: updated link transform in world frame
    xd: updated link motion in world frame
    """
    # pre-calculate some auxiliary terms used further down
    state = state.replace(i_inv=com.inv_inertia(sys, state.x))
    
    # calculate acceleration and delta-velocity terms
    tau = actuator.to_tau(sys, act, state.q, state.qd)
    xdd_i = Motion.create(vel=sys.gravity)
    xf_i = joints.resolve(sys, state, tau)
    if sys.enable_fluid:
        inertia = sys.link.inertia.i ** (1 - sys.spring_inertia_scale)
        xf_i += fluid.force(sys, state.x, state.xd, state.mass, inertia)
    
    ## <mark apply custom forces>
    #xf_i += make_ft(state, 0, state.x_i.pos[0] + jnp.array((1, 0, 0)), jnp.array((0, 0, 10)), jnp.array((0, 0, 0)))
    ## </mark>
    
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