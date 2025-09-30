import pickle
import random
import time
from typing import Any, Callable, Literal, NamedTuple, Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import logging
from brax import base, envs
from brax.training import types
from brax.v1 import envs as envs_v1
from etils import epath
from flax.struct import dataclass
from flax.training.train_state import TrainState

from jaxgcrl.envs.wrappers import TrajectoryIdWrapper
from jaxgcrl.utils.evaluator import ActorEvaluator
from jaxgcrl.utils.replay_buffer import TrajectoryUniformSamplingQueue

from .losses import update_actor_and_alpha, update_critic, update_op_actor_and_alpha, update_op_critic
from .networks import Actor, Encoder
import functools

Metrics = types.Metrics
Env = Union[envs.Env, envs_v1.Env, envs_v1.Wrapper]
State = Union[envs.State, envs_v1.State]

############################################
GCARL=True

@dataclass
class TrainingState:
    """Contains training state for the learner"""

    env_steps: jnp.ndarray
    gradient_steps: jnp.ndarray
    actor_state: TrainState
    critic_state: TrainState
    alpha_state: TrainState


class Transition(NamedTuple):
    """Container for a transition"""

    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    extras: jnp.ndarray = ()


@functools.partial(jax.jit, static_argnames=("buffer_config"))
def flatten_batch(buffer_config, transition, sample_key):

    gamma, state_size, goal_indices = buffer_config

    # Because it's vmaped transition.obs.shape is of shape (episode_len, obs_dim)
    seq_len = transition.observation.shape[0]
    arrangement = jnp.arange(seq_len)
    is_future_mask = jnp.array(
        arrangement[:, None] < arrangement[None], dtype=jnp.float32
    )  # upper triangular matrix of shape seq_len, seq_len where all non-zero entries are 1
    discount = gamma ** jnp.array(
        arrangement[None] - arrangement[:, None], dtype=jnp.float32
    )
    probs = is_future_mask * discount

    # probs is an upper triangular matrix of shape seq_len, seq_len of the form:
    #    [[0.        , 0.99      , 0.98010004, 0.970299  , 0.960596 ],
    #    [0.        , 0.        , 0.99      , 0.98010004, 0.970299  ],
    #    [0.        , 0.        , 0.        , 0.99      , 0.98010004],
    #    [0.        , 0.        , 0.        , 0.        , 0.99      ],
    #    [0.        , 0.        , 0.        , 0.        , 0.        ]]
    # assuming seq_len = 5
    # the same result can be obtained using probs = is_future_mask * (gamma ** jnp.cumsum(is_future_mask, axis=-1))

    single_trajectories = jnp.concatenate(
        [transition.extras["state_extras"]["traj_id"][:, jnp.newaxis].T] * seq_len,
        axis=0,
    )
    # array of seq_len x seq_len where a row is an array of traj_ids that correspond to the episode index from which that time-step was collected
    # timesteps collected from the same episode will have the same traj_id. All rows of the single_trajectories are same.

    probs = (
        probs * jnp.equal(single_trajectories, single_trajectories.T)
        + jnp.eye(seq_len) * 1e-5
    )
    # ith row of probs will be non zero only for time indices that
    # 1) are greater than i
    # 2) have the same traj_id as the ith time index

    goal_index = jax.random.categorical(sample_key, jnp.log(probs))
    future_state = jnp.take(
        transition.observation, goal_index[:-1], axis=0
    )  # the last goal_index cannot be considered as there is no future.
    future_action = jnp.take(transition.action, goal_index[:-1], axis=0)
    goal = future_state[:, goal_indices]
    future_state = future_state[:, :state_size]
    state = transition.observation[:-1, :state_size]  # all states are considered
    new_obs = jnp.concatenate([state, goal], axis=1)

    extras = {
        "policy_extras": {},
        "state_extras": {
            "truncation": jnp.squeeze(
                transition.extras["state_extras"]["truncation"][:-1]
            ),
            "traj_id": jnp.squeeze(transition.extras["state_extras"]["traj_id"][:-1]),
        },
        "state": state,
        "future_state": future_state,
        "future_action": future_action,
        "other_actions": transition.extras["other_actions"][:-1]

    }

    return transition._replace(
        observation=jnp.squeeze(
            new_obs
        ),  # this has shape (num_envs, episode_length-1, obs_size)
        action=jnp.squeeze(transition.action[:-1]),
        reward=jnp.squeeze(transition.reward[:-1]),
        discount=jnp.squeeze(transition.discount[:-1]),
        extras=extras,
    )


def load_params(path: str):
    with epath.Path(path).open("rb") as fin:
        buf = fin.read()
    return pickle.loads(buf)


def save_params(path: str, params: Any):
    """Saves parameters in flax format."""
    with epath.Path(path).open("wb") as fout:
        fout.write(pickle.dumps(params))


@dataclass
class CRL:
    """Contrastive Reinforcement Learning (CRL) agent."""

    policy_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 256

    # gamma
    discounting: float = 0.99
    
    ##################################
    damping: float = 0.1
    
    # forward CRL logsumexp penalty
    logsumexp_penalty_coeff: float = 0.1

    train_step_multiplier: int = 1

    disable_entropy_actor: bool = False

    max_replay_size: int = 10000
    min_replay_size: int = 1000
    unroll_length: int = 62
    h_dim: int = 256
    n_hidden: int = 2
    skip_connections: int = 4
    use_relu: bool = False

    # phi(s,a) and psi(g) repr dimension
    repr_dim: int = 64

    # layer norm
    use_ln: bool = False

    contrastive_loss_fn: Literal[
        "fwd_infonce", "sym_infonce", "bwd_infonce", "binary_nce"
    ] = "fwd_infonce"
    energy_fn: Literal["norm", "l2", "dot", "cosine"] = "norm"

    def check_config(self, config):
        """
        episode_length: the maximum length of an episode
            NOTE: `num_envs * (episode_length - 1)` must be divisible by
            `batch_size` due to the way data is stored in replay buffer.
        """
        assert (
            config.num_envs * (config.episode_length - 1) % self.batch_size == 0
        ), "num_envs * (episode_length - 1) must be divisible by batch_size"

    def train_fn(
        self,
        config: "RunConfig",
        train_env: Union[envs_v1.Env, envs.Env],
        eval_env: Optional[Union[envs_v1.Env, envs.Env]] = None,
        randomization_fn: Optional[
            Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
        ] = None,
        progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    ):

        self.check_config(config)

        unwrapped_env = train_env
        train_env = TrajectoryIdWrapper(train_env)
        train_env = envs.training.wrap(
            train_env,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
        )

        eval_env = TrajectoryIdWrapper(eval_env)
        eval_env = envs.training.wrap(
            eval_env,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
        )

        env_steps_per_actor_step = config.num_envs * self.unroll_length
        num_prefill_env_steps = self.min_replay_size * config.num_envs
        num_prefill_actor_steps = np.ceil(self.min_replay_size / self.unroll_length)
        num_training_steps_per_epoch = 40
        #(
        #    config.total_env_steps - num_prefill_env_steps
        #) // (config.num_evals * env_steps_per_actor_step)

        assert (
            num_training_steps_per_epoch > 0
        ), "total_env_steps too small for given num_envs and episode_length"

        logging.info(
            "num_prefill_env_steps: %d",
            num_prefill_env_steps,
        )
        logging.info(
            "num_prefill_actor_steps: %d",
            num_prefill_actor_steps,
        )
        logging.info(
            "num_training_steps_per_epoch: %d",
            num_training_steps_per_epoch,
        )
        ###############################################################
        if GCARL:
            random.seed(config.seed)
            np.random.seed(config.seed)
            key = jax.random.PRNGKey(config.seed)
            (
                key, op_key, buffer_key, op_buffer_key, eval_env_key, env_key, actor_key, op_actor_key, sa_key, op_sa_key,
                g_key, op_g_key
            ) = jax.random.split(key, 12)
            

            env_keys = jax.random.split(env_key, config.num_envs)
            env_state = jax.jit(train_env.reset)(env_keys)
            train_env.step = jax.jit(train_env.step)
        else:
            random.seed(config.seed)
            np.random.seed(config.seed)
            key = jax.random.PRNGKey(config.seed)
            key, buffer_key, eval_env_key, env_key, actor_key, sa_key, g_key = (
                jax.random.split(key, 7)
            )

            env_keys = jax.random.split(env_key, config.num_envs)
            env_state = jax.jit(train_env.reset)(env_keys)
            train_env.step = jax.jit(train_env.step)

        # Dimensions definitions and sanity checks
        action_size = train_env.action_size
        state_size = train_env.state_dim
        goal_size = len(train_env.goal_indices)
        obs_size = state_size + goal_size
        assert (
            obs_size == train_env.observation_size
        ), f"obs_size: {obs_size}, observation_size: {train_env.observation_size}"

        # Network setup
        # Actor
        actor = Actor(
            action_size=action_size,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
        )
        actor_state = TrainState.create(
            apply_fn=actor.apply,
            params=actor.init(actor_key, np.ones([1, obs_size])),
            tx=optax.adam(learning_rate=self.policy_lr),
        )

        ####################################################
        # Opposing Actor
        if GCARL:
            op_actor = Actor(
                action_size=action_size,
                network_width=self.h_dim,
                network_depth=self.n_hidden,
                skip_connections=self.skip_connections,
                use_relu=self.use_relu,
            )
            op_actor_state = TrainState.create(
            apply_fn=op_actor.apply,
            params=op_actor.init(op_actor_key, np.ones([1, obs_size])),
            tx=optax.adam(learning_rate=self.policy_lr),
            )

        # Critic
        sa_encoder = Encoder(
            repr_dim=self.repr_dim,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
        )
        sa_encoder_params = sa_encoder.init(
            sa_key, np.ones([1, state_size + 2*action_size])
        )
        g_encoder = Encoder(
            repr_dim=self.repr_dim,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
        )
        g_encoder_params = g_encoder.init(g_key, np.ones([1, goal_size]))
        critic_state = TrainState.create(
            apply_fn=None,
            params={"sa_encoder": sa_encoder_params, "g_encoder": g_encoder_params},
            tx=optax.adam(learning_rate=self.critic_lr),
        )
        
        ####################################################
        # Opposing Critic
        if GCARL:
            op_sa_encoder = Encoder(
                repr_dim=self.repr_dim,
                network_width=self.h_dim,
                network_depth=self.n_hidden,
                skip_connections=self.skip_connections,
                use_relu=self.use_relu,
                use_ln=self.use_ln,
            )
            op_sa_encoder_params = op_sa_encoder.init(
                op_sa_key, np.ones([1, state_size + 2*action_size])
            )
            op_g_encoder = Encoder(
                repr_dim=self.repr_dim,
                network_width=self.h_dim,
                network_depth=self.n_hidden,
                skip_connections=self.skip_connections,
                use_relu=self.use_relu,
                use_ln=self.use_ln,
            )
            op_g_encoder_params = op_g_encoder.init(op_g_key, np.ones([1, goal_size]))
            op_critic_state = TrainState.create(
                apply_fn=None,
                params={"sa_encoder": op_sa_encoder_params, "g_encoder": op_g_encoder_params},
                tx=optax.adam(learning_rate=self.critic_lr),
        )
            
        # Entropy coefficient
        target_entropy = -0.5 * action_size
        log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
        alpha_state = TrainState.create(
            apply_fn=None,
            params={"log_alpha": log_alpha},
            tx=optax.adam(learning_rate=self.alpha_lr),
        )
        
        ####################################################
        # Op Entropy Coefficient
        op_target_entropy = -0.5 * action_size
        op_log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
        op_alpha_state = TrainState.create(
            apply_fn=None,
            params={"log_alpha": op_log_alpha},
            tx=optax.adam(learning_rate=self.alpha_lr),
        )

        # Trainstate
        training_state = TrainingState(
            env_steps=jnp.zeros(()),
            gradient_steps=jnp.zeros(()),
            actor_state=actor_state,
            critic_state=critic_state,
            alpha_state=alpha_state,
        )
        ##############################################
        # Op Trainstate
        op_training_state = TrainingState(
            env_steps=jnp.zeros(()),
            gradient_steps=jnp.zeros(()),
            actor_state=op_actor_state,
            critic_state=op_critic_state,
            alpha_state=op_alpha_state,
        )

        # Replay Buffer
        dummy_obs = jnp.zeros((obs_size,))
        dummy_action = jnp.zeros((action_size,))

        dummy_transition = Transition(
            observation=dummy_obs,
            action=dummy_action,
            reward=0.0,
            discount=0.0,
            extras={
                "state_extras": {
                    "truncation": 0.0,
                    "traj_id": 0.0,
                },
                "other_actions": jnp.zeros_like(dummy_action),
            },
        )
       

        def jit_wrap(buffer):
            buffer.insert_internal = jax.jit(buffer.insert_internal)
            buffer.sample_internal = jax.jit(buffer.sample_internal)
            return buffer

        replay_buffer = jit_wrap(
            TrajectoryUniformSamplingQueue(
                max_replay_size=self.max_replay_size,
                dummy_data_sample=dummy_transition,
                sample_batch_size=self.batch_size,
                num_envs=config.num_envs,
                episode_length=config.episode_length,
            )
        )
        buffer_state = jax.jit(replay_buffer.init)(buffer_key)

        ####################################################
        if GCARL:
            op_replay_buffer = jit_wrap(
                TrajectoryUniformSamplingQueue(
                    max_replay_size=self.max_replay_size,
                    dummy_data_sample=dummy_transition,
                    sample_batch_size=self.batch_size,
                    num_envs=config.num_envs,
                    episode_length=config.episode_length,
                )
            )
            op_buffer_state = jax.jit(replay_buffer.init)(buffer_key)

        ########################################################
        # GCARL Deterministic Actor Step
        #if GCARL:
         #   def deterministic_actor_step(training_state, op_training_state, env, env_state, extra_fields):
           #     means, _ = actor.apply(training_state.actor_state.params, env_state.obs)
             #   op_means, _ = op_actor.apply(op_training_state.actor_state.params)
             #   actions = nn.tanh(means)
             #   op_actions = self.damping*nn.tanh(op_means)
             #   net_action = actions + op_actions

               # nstate = env.step(env_state, net_action)
              #  state_extras = {x: nstate.info[x] for x in extra_fields}

             #   return nstate, Transition(
              #      observation=env_state.obs,
              #      action=actions,
              #      reward=nstate.reward,
               #     discount=1 - nstate.done,
                #    extras={"state_extras": state_extras,
                #            "other_actions" : op_actions    },
                #)
        #else:
        def deterministic_actor_step(training_state, env, env_state, extra_fields):
            means, _ = actor.apply(training_state.actor_state.params, env_state.obs)
            actions = nn.tanh(means)
    
            nstate = env.step(env_state, actions)
            state_extras = {x: nstate.info[x] for x in extra_fields}
    
            return nstate, Transition(
                observation=env_state.obs,
                action=actions,
                reward=nstate.reward,
                discount=1 - nstate.done,
                extras={"state_extras": state_extras},
            )
        ##########################################################
        # GCARL Actor Step
        if GCARL:
            def actor_step(actor_state, op_actor_state, env, env_state, key, extra_fields):
                means, log_stds = actor.apply(actor_state.params, env_state.obs)
                op_means, _ = op_actor.apply(op_actor_state.params, env_state.obs)
                stds = jnp.exp(log_stds)
                actions = nn.tanh(
                    means
                    + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype)
                )
                op_actions = self.damping*nn.tanh(op_means)
                net_action = actions + op_actions
                
                nstate = env.step(env_state, net_action)
                state_extras = {x: nstate.info[x] for x in extra_fields}

                return nstate, Transition(
                    observation=env_state.obs,
                    action=actions,
                    reward=nstate.reward,
                    discount=1 - nstate.done,
                    extras={"state_extras": state_extras,
                            "other_actions" : op_actions    },
                )
        else:
            def actor_step(actor_state, env, env_state, key, extra_fields):
                means, log_stds = actor.apply(actor_state.params, env_state.obs)
                stds = jnp.exp(log_stds)
                actions = nn.tanh(
                    means
                    + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype)
                )
    
                nstate = env.step(env_state, actions)
                state_extras = {x: nstate.info[x] for x in extra_fields}
    
                return nstate, Transition(
                    observation=env_state.obs,
                    action=actions,
                    reward=nstate.reward,
                    discount=1 - nstate.done,
                    extras={"state_extras": state_extras},
                )
                
        ########################################################
        # GCARL Deterministic Op Actor Step
        if GCARL:
            def deterministic_op_actor_step(training_state, env, env_state, extra_fields):
                means, _ = actor.apply(training_state.actor_state.params, env_state.obs)
                op_means, _ = op_actor.apply(training_state.actor_state)
                actions = nn.tanh(means)
                op_actions = self.damping*nn.tanh(op_means)
                net_action = actions + op_actions

                nstate = env.step(env_state, net_action)
                state_extras = {x: nstate.info[x] for x in extra_fields}

                return nstate, Transition(
                    observation=env_state.obs,
                    action=op_actions,
                    reward=nstate.reward,
                    discount=1 - nstate.done,
                    extras={"state_extras": state_extras,
                            "other_actions" : actions          },
                )
                
        ##########################################################
        # GCARL Op Actor Step
        if GCARL:
            def op_actor_step(actor_state, op_actor_state, env, env_state, key, extra_fields):
                means, _ = actor.apply(actor_state.params, env_state.obs)
                op_means, op_log_stds = op_actor.apply(op_actor_state.params, env_state.obs)
                op_stds = jnp.exp(op_log_stds)
                op_actions = self.damping* nn.tanh(
                    op_means
                    + op_stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype)
                )
                actions = nn.tanh(means)
                net_action = actions + op_actions
                
                nstate = env.step(env_state, net_action)
                state_extras = {x: nstate.info[x] for x in extra_fields}

                return nstate, Transition(
                    observation=env_state.obs,
                    action=op_actions,
                    reward=nstate.reward,
                    discount=1 - nstate.done,
                    extras={"state_extras": state_extras,
                            "other_actions" : actions    },
                )

    
            
        ################################################
        if GCARL:
            def get_experience(actor_state, op_actor_state, env_state, buffer_state, key):
                @jax.jit
                def f(carry, unused_t):
                    env_state, current_key = carry
                    current_key, next_key = jax.random.split(current_key)
                    env_state, transition = actor_step(
                        actor_state,
                        op_actor_state,
                        train_env,
                        env_state,
                        current_key,
                        extra_fields=("truncation", "traj_id"),
                    )
                    return (env_state, next_key), transition

                (env_state, _), data = jax.lax.scan(
                    f, (env_state, key), (), length=self.unroll_length
                )


                buffer_state = replay_buffer.insert(buffer_state, data)
                return env_state, buffer_state

            def get_op_experience(actor_state, op_actor_state, env_state, buffer_state, key):
                @jax.jit
                def f(carry, unused_t):
                    env_state, current_key = carry
                    current_key, next_key = jax.random.split(current_key)
                    env_state, transition = op_actor_step(
                        actor_state,
                        op_actor_state,
                        train_env,
                        env_state,
                        current_key,
                        extra_fields=("truncation", "traj_id"),
                    )
                    return (env_state, next_key), transition

                (env_state, _), data = jax.lax.scan(
                    f, (env_state, key), (), length=self.unroll_length
                )

                buffer_state = replay_buffer.insert(buffer_state, data)
                return env_state, buffer_state
        else:
            @jax.jit
            def get_experience(actor_state, env_state, buffer_state, key):
                @jax.jit
                def f(carry, unused_t):
                    env_state, current_key = carry
                    current_key, next_key = jax.random.split(current_key)
                    env_state, transition = actor_step(
                        actor_state,
                        train_env,
                        env_state,
                        current_key,
                        extra_fields=("truncation", "traj_id"),
                    )
                    return (env_state, next_key), transition
    
                (env_state, _), data = jax.lax.scan(
                    f, (env_state, key), (), length=self.unroll_length
                )
    
                buffer_state = replay_buffer.insert(buffer_state, data)
                return env_state, buffer_state


        ########################################################
        if GCARL:
            def prefill_replay_buffer(training_state, op_training_state, env_state, buffer_state, key):
                @jax.jit
                def f(carry, unused):
                    del unused
                    training_state, env_state, buffer_state, key = carry
                    key, new_key = jax.random.split(key)
                    env_state, buffer_state = get_experience(
                        training_state.actor_state,
                        op_training_state.actor_state,
                        env_state,
                        buffer_state,
                        key,
                    )
                    training_state = training_state.replace(
                        env_steps=training_state.env_steps + env_steps_per_actor_step,
                    )
                    return (training_state, env_state, buffer_state, new_key), ()

                return jax.lax.scan(
                    f,
                    (training_state, env_state, buffer_state, key),
                    (),
                    length=num_prefill_actor_steps,
                )[0]
                
            def prefill_op_replay_buffer(training_state, op_training_state, env_state, buffer_state, key):
                @jax.jit
                def f(carry, unused):
                    del unused
                    op_training_state, env_state, buffer_state, key = carry
                    key, new_key = jax.random.split(key)
                    env_state, buffer_state = get_op_experience(
                        training_state.actor_state,
                        op_training_state.actor_state,
                        env_state,
                        buffer_state,
                        key,
                    )
                    op_training_state = op_training_state.replace(
                        env_steps=op_training_state.env_steps + env_steps_per_actor_step,
                    )
                    return (op_training_state, env_state, buffer_state, new_key), ()

                return jax.lax.scan(
                    f,
                    (op_training_state, env_state, buffer_state, key),
                    (),
                    length=num_prefill_actor_steps,
                )[0]
        else:
            def prefill_replay_buffer(training_state, env_state, buffer_state, key):
                @jax.jit
                def f(carry, unused):
                    del unused
                    training_state, env_state, buffer_state, key = carry
                    key, new_key = jax.random.split(key)
                    env_state, buffer_state = get_experience(
                        training_state.actor_state,
                        env_state,
                        buffer_state,
                        key,
                    )
                    training_state = training_state.replace(
                        env_steps=training_state.env_steps + env_steps_per_actor_step,
                    )
                    return (training_state, env_state, buffer_state, new_key), ()
    
                return jax.lax.scan(
                    f,
                    (training_state, env_state, buffer_state, key),
                    (),
                    length=num_prefill_actor_steps,
                )[0]

        @jax.jit
        def update_networks(carry, transitions):
            training_state, key = carry
            key, critic_key, actor_key = jax.random.split(key, 3)

            context = dict(
                **vars(self),
                **vars(config),
                state_size=state_size,
                action_size=action_size,
                goal_size=goal_size,
                obs_size=obs_size,
                goal_indices=train_env.goal_indices,
                target_entropy=target_entropy,
            )

            networks = dict(
                actor=actor,
                sa_encoder=sa_encoder,
                g_encoder=g_encoder,
            )

            training_state, actor_metrics = update_actor_and_alpha(
                context, networks, transitions, training_state, actor_key
            )
            training_state, critic_metrics = update_critic(
                context, networks, transitions, training_state, critic_key
            )
            training_state = training_state.replace(
                gradient_steps=training_state.gradient_steps + 1
            )

            metrics = {}
            metrics.update(actor_metrics)
            metrics.update(critic_metrics)

            return (
                training_state,
                key,
            ), metrics

        ######################################################
        if GCARL:
            @jax.jit
            def update_op_networks(carry, transitions):
                training_state, key = carry
                op_key, op_critic_key, op_actor_key = jax.random.split(key, 3)

                context = dict(
                    **vars(self),
                    **vars(config),
                    state_size=state_size,
                    action_size=action_size,
                    goal_size=goal_size,
                    obs_size=obs_size,
                    goal_indices=train_env.goal_indices,
                    target_entropy=target_entropy,
                    op_target_entropy=op_target_entropy,
                )

                networks = dict(
                    actor=op_actor,
                    sa_encoder=op_sa_encoder,
                    g_encoder=op_g_encoder,
                )

                op_training_state, op_actor_metrics = update_op_actor_and_alpha(
                    context, networks, transitions, training_state, op_actor_key
                )
                op_training_state, op_critic_metrics = update_op_critic(
                    context, networks, transitions, training_state, op_critic_key
                )
                op_training_state = op_training_state.replace(
                    gradient_steps=op_training_state.gradient_steps + 1
                )

                metrics = {}
                metrics.update(op_actor_metrics)
                metrics.update(op_critic_metrics)

                return (
                    training_state,
                    key,
                ), metrics

        ###########################################################
        if GCARL:
            @jax.jit
            def training_step(training_state, op_training_state, env_state, buffer_state, key):
                experience_key1, experience_key2, sampling_key, training_key = (
                    jax.random.split(key, 4)
                )

                # update buffer
                env_state, buffer_state = get_experience(
                    training_state.actor_state,
                    op_training_state.actor_state,
                    env_state,
                    buffer_state,
                    experience_key1,
                )

                training_state = training_state.replace(
                    env_steps=training_state.env_steps + env_steps_per_actor_step,
                )

                # sample actor-step worth of transitions
                buffer_state, transitions = replay_buffer.sample(buffer_state)

                # process transitions for training
                batch_keys = jax.random.split(
                    sampling_key, transitions.observation.shape[0]
                )
                transitions = jax.vmap(flatten_batch, in_axes=(None, 0, 0))(
                    (self.discounting, state_size, tuple(train_env.goal_indices)),
                    transitions,
                    batch_keys,
                )
                transitions = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"), transitions
                )

                # permute transitions
                permutation = jax.random.permutation(
                    experience_key2, len(transitions.observation)
                )
                transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)
                transitions = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, (-1, self.batch_size) + x.shape[1:]),
                    transitions,
                )

                # take actor-step worth of training-step
                (
                    training_state,
                    _,
                ), metrics = jax.lax.scan(
                    update_networks, (training_state, training_key), transitions
                )

                return (
                    training_state,
                    env_state,
                    buffer_state,
                ), metrics

            @jax.jit
            def op_training_step(training_state, op_training_state, env_state, buffer_state, key):
                op_experience_key1, op_experience_key2, op_sampling_key, op_training_key = (
                    jax.random.split(key, 4)
                )

                # update buffer
                env_state, buffer_state = get_op_experience(
                    training_state.actor_state,
                    op_training_state.actor_state,
                    env_state,
                    buffer_state,
                    op_experience_key1,
                )

                op_training_state = op_training_state.replace(
                    env_steps=op_training_state.env_steps + env_steps_per_actor_step,
                )

                # sample actor-step worth of transitions
                buffer_state, transitions = op_replay_buffer.sample(buffer_state)

                # process transitions for training
                op_batch_keys = jax.random.split(
                    op_sampling_key, transitions.observation.shape[0]
                )
                transitions = jax.vmap(flatten_batch, in_axes=(None, 0, 0))(
                    (self.discounting, state_size, tuple(train_env.goal_indices)),
                    transitions,
                    op_batch_keys,
                )
                transitions = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"), transitions
                )

                # permute transitions
                permutation = jax.random.permutation(
                    op_experience_key2, len(transitions.observation)
                )
                transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)
                transitions = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, (-1, self.batch_size) + x.shape[1:]),
                    transitions,
                )

                # take actor-step worth of training-step
                (
                    op_training_state,
                    _,
                ), metrics = jax.lax.scan(
                    update_op_networks, (op_training_state, op_training_key), transitions
                )

                return (
                    op_training_state,
                    env_state,
                    buffer_state,
                ), metrics
        else:
            @jax.jit
            def training_step(training_state, env_state, buffer_state, key):
                experience_key1, experience_key2, sampling_key, training_key = (
                    jax.random.split(key, 4)
                )
    
                # update buffer
                env_state, buffer_state = get_experience(
                    training_state.actor_state,
                    env_state,
                    buffer_state,
                    experience_key1,
                )
    
                training_state = training_state.replace(
                    env_steps=training_state.env_steps + env_steps_per_actor_step,
                )
    
                # sample actor-step worth of transitions
                buffer_state, transitions = replay_buffer.sample(buffer_state)
    
                # process transitions for training
                batch_keys = jax.random.split(
                    sampling_key, transitions.observation.shape[0]
                )
                transitions = jax.vmap(flatten_batch, in_axes=(None, 0, 0))(
                    (self.discounting, state_size, tuple(train_env.goal_indices)),
                    transitions,
                    batch_keys,
                )
                transitions = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"), transitions
                )
    
                # permute transitions
                permutation = jax.random.permutation(
                    experience_key2, len(transitions.observation)
                )
                transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)
                transitions = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, (-1, self.batch_size) + x.shape[1:]),
                    transitions,
                )
    
                # take actor-step worth of training-step
                (
                    training_state,
                    _,
                ), metrics = jax.lax.scan(
                    update_networks, (training_state, training_key), transitions
                )
    
                return (
                    training_state,
                    env_state,
                    buffer_state,
                ), metrics

        ##############################################################
        if GCARL:
            #@jax.jit
            def training_epoch(
                training_state,
                op_training_state,
                env_state,
                buffer_state,
                op_buffer_state,
                key,
            ):
                print("Training epoch: entered")
                #@jax.jit
                def f(carry, unused_t):
                    ts, op_ts, es, bs, op_bs, k = carry
                    k, train_key = jax.random.split(k, 2)
                    print("training_step: before call")
                    try:
                        
                        (
                            ts,
                            es,
                            bs,
                        ), metrics = training_step(ts, op_ts, es, bs, train_key)
                    except Exception as e:
                        print(f"training_step: exception - {e}")
                    return (ts, op_ts, es, bs, op_bs, k), metrics

                #@jax.jit
                def g(carry, unused_t):
                    ts, op_ts, es, bs, op_bs, k = carry
                    k, train_key = jax.random.split(k, 2)
                    print("Training epoch: about to call op_training_step")
                    (
                        op_ts,
                        es,
                        op_bs,
                    ), op_metrics = op_training_step(ts, op_ts, es, op_bs, train_key)
                    print("Training epoch: op_training_step returned")
                    return (ts, op_ts, es, bs, op_bs, k), op_metrics

                print("Training epoch: about to scan training_step")

                (
                    (training_state, op_training_state, env_state, buffer_state, op_buffer_state, key), metrics
                ) = jax.lax.scan(
                    f,
                    (training_state, op_training_state, env_state, buffer_state, op_buffer_state, key),
                    (),
                    length=num_training_steps_per_epoch,
                )

                (
                    (training_state, op_training_state, env_state, buffer_state, op_buffer_state, key), op_metrics
                ) = jax.lax.scan(
                    g,
                    (training_state, op_training_state, env_state, buffer_state, op_buffer_state, key),
                    (),
                    length=num_training_steps_per_epoch,
                )

                metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
                op_metrics["buffer_current_size"] = op_replay_buffer.size(op_buffer_state)
                return training_state, op_training_state, env_state, buffer_state, op_buffer_state, metrics, op_metrics
        else:
            @jax.jit
            def training_epoch(
                training_state,
                env_state,
                buffer_state,
                key,
            ):
                @jax.jit
                def f(carry, unused_t):
                    ts, es, bs, k = carry
                    k, train_key = jax.random.split(k, 2)
                    (
                        ts,
                        es,
                        bs,
                    ), metrics = training_step(ts, es, bs, train_key)
                    return (ts, es, bs, k), metrics
    
                (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(
                    f,
                    (training_state, env_state, buffer_state, key),
                    (),
                    length=num_training_steps_per_epoch,
                )
    
                metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
                return training_state, env_state, buffer_state, metrics

        
        key, prefill_key, op_prefill_key = jax.random.split(key, 3)

        training_state, env_state, buffer_state, _ = prefill_replay_buffer(
            training_state, op_training_state, env_state, buffer_state, prefill_key
        )

        op_training_state, env_state, buffer_state, _ = prefill_op_replay_buffer(
            training_state, op_training_state, env_state, op_buffer_state, op_prefill_key
        )

        """Setting up evaluator"""
        evaluator = ActorEvaluator(
            deterministic_actor_step,
            eval_env,
            num_eval_envs=config.num_eval_envs,
            episode_length=config.episode_length,
            key=eval_env_key,
        )

        training_walltime = 0
        logging.info("starting training....")
        for ne in range(config.num_evals):
            print(f"Entered training loop at eval {ne}")
            logging.info(f"Entered training loop at eval {ne}")

            t = time.time()

            key, epoch_key = jax.random.split(key)
            try:
                (
                    training_state, op_training_state, env_state, buffer_state, op_buffer_state, metrics, op_metrics
                ) = training_epoch(
                    training_state, op_training_state, env_state, buffer_state, op_buffer_state, epoch_key
                )
            except Exception as e:
                logging.error(f"Error in training_epoch: {e}", exc_info=True)
                raise
            print("After training epoch")

            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            metrics = jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

            op_metrics = jax.tree_util.tree_map(jnp.mean, op_metrics)
            op_metrics = jax.tree_util.tree_map(lambda x: x.block_until_ready(), op_metrics)

            epoch_training_time = time.time() - t
            training_walltime += epoch_training_time

            sps = (
                env_steps_per_actor_step * num_training_steps_per_epoch
            ) / epoch_training_time
            metrics = {
                "training/sps": sps,
                "training/walltime": training_walltime,
                "training/envsteps": training_state.env_steps.item(),
                **{f"training/{name}": value for name, value in metrics.items()},
            }
            current_step = int(training_state.env_steps.item())
            try:
                metrics = evaluator.run_evaluation(training_state, metrics)
            except Exception as e:
                logging.error(f"Error in evaluator: {e}", exc_info=True)
                raise
            print("After evaluator")
            logging.info("step: %d", current_step)

            do_render = ne % config.visualization_interval == 0
            make_policy = lambda param: lambda obs, rng: actor.apply(param, obs)
            try:
                progress_fn(
                    current_step,
                    metrics,
                    make_policy,
                    training_state.actor_state.params,
                    unwrapped_env,
                    do_render=do_render,
                )
            except Exception as e:
                logging.error(f"Error in progress_fn: {e}", exc_info=True)
                raise
            print("After progress_fn")

            if config.checkpoint_logdir:
                # Save current policy and critic params.
                params = (
                    training_state.alpha_state.params,
                    training_state.actor_state.params,
                    training_state.critic_state.params,
                )
                path = f"{config.checkpoint_logdir}/step_{int(training_state.env_steps)}.pkl"
                save_params(path, params)

        total_steps = current_step
        assert total_steps >= config.total_env_steps

        logging.info("total steps: %s", total_steps)

        return make_policy, params, metrics
