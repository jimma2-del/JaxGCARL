import functools
import logging
import pickle
import random
import time
from typing import Any, Callable, Literal, NamedTuple, Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax import base, envs
from brax.training import types
from brax.v1 import envs as envs_v1
from etils import epath
from flax.struct import dataclass
from flax.training.train_state import TrainState

from jaxgcrl.envs.wrappers import TrajectoryIdWrapper
from jaxgcrl.utils.evaluator import ActorEvaluator
from jaxgcrl.utils.replay_buffer import TrajectoryUniformSamplingQueue

from .losses import update_actor_and_alpha, update_critic # MARK 1
from .networks import Actor, Encoder

Metrics = types.Metrics
Env = Union[envs.Env, envs_v1.Env, envs_v1.Wrapper]
State = Union[envs.State, envs_v1.State]

# Unifying functions branch

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
    protag_action: jnp.ndarray
    antag_action: jnp.ndarray
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
    discount = gamma ** jnp.array(arrangement[None] - arrangement[:, None], dtype=jnp.float32)
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

    probs = probs * jnp.equal(single_trajectories, single_trajectories.T) + jnp.eye(seq_len) * 1e-5
    # ith row of probs will be non zero only for time indices that
    # 1) are greater than i
    # 2) have the same traj_id as the ith time index

    goal_index = jax.random.categorical(sample_key, jnp.log(probs))
    future_state = jnp.take(
        transition.observation, goal_index[:-1], axis=0
    )  # the last goal_index cannot be considered as there is no future.
    future_protag_action = jnp.take(transition.protag_action, goal_index[:-1], axis=0)
    future_antag_action = jnp.take(transition.antag_action, goal_index[:-1], axis=0) # added antag, switched future_action to future_protag_...
    goal = future_state[:, goal_indices]
    future_state = future_state[:, :state_size]
    state = transition.observation[:-1, :state_size]  # all states are considered
    new_obs = jnp.concatenate([state, goal], axis=1)

    extras = {
        "policy_extras": {},
        "state_extras": {
            "truncation": jnp.squeeze(transition.extras["state_extras"]["truncation"][:-1]),
            "traj_id": jnp.squeeze(transition.extras["state_extras"]["traj_id"][:-1]),
        },
        "state": state,
        "future_state": future_state,
        "future_protag_action": future_protag_action,
        "future_antag_action": future_antag_action # mark
    }

    return transition._replace(
        observation=jnp.squeeze(new_obs),  # this has shape (num_envs, episode_length-1, obs_size)
        protag_action=jnp.squeeze(transition.protag_action[:-1]),
        antag_action=jnp.squeeze(transition.antag_action[:-1]), # mark
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
class CRL: #mark CARL, but keeping CRL for now for convenience #TODO
    """Contrastive Adversarial Reinforcement Learning (CARL) agent."""

    policy_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 256

    # gamma
    discounting: float = 0.99

    # damping, mark
    antag_damping: float = 0.1

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

    contrastive_loss_fn: Literal["fwd_infonce", "sym_infonce", "bwd_infonce", "binary_nce"] = "fwd_infonce"
    energy_fn: Literal["norm", "l2", "dot", "cosine"] = "norm"

    def check_config(self, config):
        """
        episode_length: the maximum length of an episode
            NOTE: `num_envs * (episode_length - 1)` must be divisible by
            `batch_size` due to the way data is stored in replay buffer.
        """
        assert config.num_envs * (config.episode_length - 1) % self.batch_size == 0, (
            "num_envs * (episode_length - 1) must be divisible by batch_size"
        )

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

        env_steps_per_actor_step = config.num_envs * self.unroll_length # mark leaving the same, assuming it is equivalent for both protag & antag
        num_prefill_env_steps = self.min_replay_size * config.num_envs
        num_prefill_actor_steps = np.ceil(self.min_replay_size / self.unroll_length) # mark leaving the same for both
        num_training_steps_per_epoch = (config.total_env_steps - num_prefill_env_steps) // (
            config.num_evals * env_steps_per_actor_step
        )

        assert num_training_steps_per_epoch > 0, (
            "total_env_steps too small for given num_envs and episode_length"
        )

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

        random.seed(config.seed)
        np.random.seed(config.seed)
        key = jax.random.PRNGKey(config.seed)
        # mark: Not using 2 buffers or 2 env type keys or 2 main keys, but otherwise duplicating keys
        (
            primary_key, buffer_key, eval_env_key, env_key, protag_actor_key, antag_actor_key, protag_sa_key, antag_sa_key,
            protag_g_key, antag_g_key
        ) = jax.random.split(key, 10)

        env_keys = jax.random.split(env_key, config.num_envs)
        env_state = train_env.reset(env_keys) # mark
        train_env.step = train_env.step

        # Dimensions definitions and sanity checks
        protag_action_size = train_env.action_size # MARK
        antag_action_size = train_env.action_size # MARK
        state_size = train_env.state_dim # mark, not actually changing this
        goal_size = len(train_env.goal_indices)
        obs_size = state_size + goal_size
        assert obs_size == train_env.observation_size, (
            f"obs_size: {obs_size}, observation_size: {train_env.observation_size}"
        )

        ### Network setup
        

        # Actor - mark, making general functions for actor, actor state
        def gen_actor(action_size, actor_key):
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
            return actor, actor_state

        protag_actor, protag_actor_state = gen_actor(protag_action_size, protag_actor_key)
        antag_actor, antag_actor_state = gen_actor(antag_action_size, antag_actor_key)

        # Critic - mark, making general funcs for sa, g, and critic_state
        def gen_encoders(sa_key, g_key, goal_size, state_size, action_size):
            sa_encoder = Encoder(
                repr_dim=self.repr_dim,
                network_width=self.h_dim,
                network_depth=self.n_hidden,
                skip_connections=self.skip_connections,
                use_relu=self.use_relu,
                use_ln=self.use_ln,
            )
            sa_encoder_params = sa_encoder.init(sa_key, np.ones([1, state_size + action_size])) # mark, previously had 2 times action_size
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

            return sa_encoder, g_encoder, critic_state
            
        protag_sa_encoder, protag_g_encoder, protag_critic_state = gen_encoders(protag_sa_key, protag_g_key, goal_size, state_size, protag_action_size)
        antag_sa_encoder, antag_g_encoder, antag_critic_state = gen_encoders(antag_sa_key, antag_g_key, goal_size, state_size, antag_action_size)
        

        # Entropy coefficient
        # mark general function
        def gen_entropy(action_size):
            target_entropy = -0.5 * action_size
            log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
            alpha_state = TrainState.create(
                apply_fn=None,
                params={"log_alpha": log_alpha},
                tx=optax.adam(learning_rate=self.alpha_lr),
            )
            return target_entropy, log_alpha, alpha_state

        protag_target_entropy, protag_log_alpha, protag_alpha_state = gen_entropy(protag_action_size)
        antag_target_entropy, antag_log_alpha, antag_alpha_state = gen_entropy(antag_action_size)
        

        # Trainstate
        #mark general function
        def gen_training_state(actor_state, critic_state, alpha_state):
            training_state = TrainingState(
                env_steps=jnp.zeros(()),
                gradient_steps=jnp.zeros(()),
                actor_state=actor_state,
                critic_state=critic_state,
                alpha_state=alpha_state,
            )

            return training_state

        protag_training_state = gen_training_state(protag_actor_state, protag_critic_state, protag_alpha_state)
        antag_training_state = gen_training_state(antag_actor_state, antag_critic_state, antag_alpha_state)
        
        # Replay Buffer
        dummy_obs = jnp.zeros((obs_size,))
        dummy_protag_action = jnp.zeros((protag_action_size,)) #mark protag & antag
        dummy_antag_action = jnp.zeros((antag_action_size,))

        dummy_transition = Transition(
            observation=dummy_obs,
            protag_action=dummy_protag_action, # mark protag & antag
            antag_action=dummy_antag_action,
            reward=0.0,
            discount=0.0,
            extras={
                "state_extras": {
                    "truncation": 0.0,
                    "traj_id": 0.0,
                }
            },
        )

        # mark removed jit_wrap(buffer) func since this whole thing is wrapped in jax.jit
        
        # mark removed jit_wrap around trajectoryuni...
        replay_buffer = TrajectoryUniformSamplingQueue(
                max_replay_size=self.max_replay_size,
                dummy_data_sample=dummy_transition,
                sample_batch_size=self.batch_size,
                num_envs=config.num_envs,
                episode_length=config.episode_length,
            )
        buffer_state = replay_buffer.init(buffer_key) # mark removed jax.jit

        # mark, theoretically an op/antag_buffer could exist here


        #mark, NOT making generalized bc its only called for the protagonist by ActorEvaluator. Changed actor to specifically protag_actor within only this function
        def deterministic_actor_step(training_state, env, env_state, extra_fields): 
            means, _ = protag_actor.apply(training_state.actor_state.params, env_state.obs) # mark
            actions = nn.tanh(means)

            nstate = env.step(env_state, actions)
            state_extras = {x: nstate.info[x] for x in extra_fields}

            return nstate, Transition(
                observation=env_state.obs,
                protag_action=actions,
                antag_action=None, # mark
                reward=nstate.reward,
                discount=1 - nstate.done,
                extras={"state_extras": state_extras},
            )

        # mark added actor & antag actor
        def actor_step(protag_actor_state, antag_actor_state, env, env_state, key, antag_key, extra_fields):
            
            means, log_stds = protag_actor.apply(protag_actor_state.params, env_state.obs)
            stds = jnp.exp(log_stds)
            protag_actions = nn.tanh(means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype))

            antag_means, antag_log_stds = antag_actor.apply(antag_actor_state.params, env_state.obs) # mark added log_stds for both actors
            antag_stds = jnp.exp(antag_log_stds)
            antag_actions = nn.tanh(antag_means + antag_stds * jax.random.normal(antag_key, shape=antag_means.shape, dtype=antag_means.dtype)) # mark removed damping, should be placed in net_action

            net_action = protag_actions + antag_actions # TODO edit net_action formation for forces, perhaps
            
            nstate = env.step(env_state, net_action)
            state_extras = {x: nstate.info[x] for x in extra_fields}

            return nstate, Transition( # mark transition now has antag_action
                observation=env_state.obs,
                protag_action=protag_actions,
                antag_action=antag_actions,
                reward=nstate.reward,
                discount=1 - nstate.done,
                extras={"state_extras": state_extras},
            )

        # mark protag, antag added. Modified for actor step
        @jax.jit
        def get_experience(protag_actor_state, antag_actor_state, env_state, buffer_state, key):
            
            @jax.jit
            def f(carry, unused_t):
                env_state, current_key = carry
                protag_current_key, antag_current_key, next_key = jax.random.split(current_key, 3) # splits into 3 keys now mark
                env_state, transition = actor_step(
                    protag_actor_state,
                    antag_actor_state,
                    train_env,
                    env_state,
                    protag_current_key,
                    antag_current_key,
                    extra_fields=("truncation", "traj_id"),
                )
                return (env_state, next_key), transition

            (env_state, _), data = jax.lax.scan(f, (env_state, key), (), length=self.unroll_length)

            buffer_state = replay_buffer.insert(buffer_state, data)
            return env_state, buffer_state

        # mark adjusted parameters
        def prefill_replay_buffer(protag_training_state_ext, antag_training_state_ext, env_state_ext, buffer_state_ext, key_ext):
            
            
            @jax.jit
            def f(carry, unused):
                del unused
                protag_training_state, antag_training_state, env_state, buffer_state, key = carry
                key, new_key = jax.random.split(key)
                env_state, buffer_state = get_experience( # mark adjusted arguments
                    protag_training_state.actor_state,
                    antag_training_state.actor_state,
                    env_state,
                    buffer_state,
                    key,
                )
                protag_training_state = protag_training_state.replace( # mark
                    env_steps=protag_training_state.env_steps + env_steps_per_actor_step,
                )
                antag_training_state = antag_training_state.replace( # mark doing both protag & antag at once, should be fine since we are just prefilling the replay buffer
                    env_steps=antag_training_state.env_steps + env_steps_per_actor_step,
                )
                return (protag_training_state, antag_training_state, env_state, buffer_state, new_key), () # mark returns both training states
        
            return jax.lax.scan(
                f,
                (protag_training_state_ext, antag_training_state_ext, env_state_ext, buffer_state_ext, key_ext), # mark
                (),
                length=num_prefill_actor_steps,
            )[0]

        
        # mark added negative reward flag since it is needed to calculate gradients in updte_act_and_alpha
        @functools.partial(jax.jit, static_argnames=("is_antag_flag")) 
        def update_networks(action_size, training_state_ext, training_key, transitions_ext, is_antag_flag):


            def f(carry, transitions): # Mark, creating f in this function and moving jax.lax.scan out of training_step and into update_networks
                training_state, key = carry
                key, critic_key, actor_key = jax.random.split(key, 3)

                
                target_entropy = antag_target_entropy if is_antag_flag else protag_target_entropy #mark
                
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
        
                actor = antag_actor if is_antag_flag else protag_actor # mark, cannot pass as parameter
                sa_encoder = antag_sa_encoder if is_antag_flag else protag_sa_encoder # mark, cannot pass as parameter
                g_encoder = antag_g_encoder if is_antag_flag else protag_g_encoder # mark, cannot pass as parameter
                
                networks = dict(
                    actor=actor,
                    sa_encoder=sa_encoder,
                    g_encoder=g_encoder,
                )
    
                training_state, actor_metrics = update_actor_and_alpha(
                    context, networks, transitions, training_state, actor_key, is_antag_flag, # mark negative reward flag boolean
                )
                training_state, critic_metrics = update_critic(
                    context, networks, transitions, training_state, critic_key, is_antag_flag,
                )
                training_state = training_state.replace(gradient_steps=training_state.gradient_steps + 1)
    
                metrics = {}
                metrics.update(actor_metrics)
                metrics.update(critic_metrics)
    
                return (
                    training_state,
                    key,
                ), metrics

                
            return jax.lax.scan(f, (training_state_ext, training_key), transitions_ext)

        @functools.partial(jax.jit, static_argnames=("is_antag_flag")) 
        def training_step(protag_actor_state, antag_actor_state, action_size, is_antag_flag, training_state, env_state, buffer_state, key): 
                                        # mark: Function requires both protag & antag for get exp
                                        #           These are only used for getting experience 
                                        # The following actor (a duplicate of either protag or antag_actor), along with its associated
                        # sa,g_encoders and target_entropy and action size and negativeRewardFlag is what is actually updated/trained
            experience_key1, experience_key2, sampling_key, training_key = jax.random.split(key, 4)

            # update buffer
            env_state, buffer_state = get_experience(
                protag_actor_state, 
                antag_actor_state, # added actor data needed for getting experience
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
            batch_keys = jax.random.split(sampling_key, transitions.observation.shape[0])
            transitions = jax.vmap(flatten_batch, in_axes=(None, 0, 0))(
                (self.discounting, state_size, tuple(train_env.goal_indices)),
                transitions,
                batch_keys,
            )
            transitions = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"), transitions
            )

            # permute transitions
            permutation = jax.random.permutation(experience_key2, len(transitions.observation))
            transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)
            transitions = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1, self.batch_size) + x.shape[1:]),
                transitions,
            )

            # take actor-step worth of training-step
            (
                (
                    training_state,
                    _,
                ),
                metrics,
            ) = update_networks(action_size, training_state, training_key, transitions, is_antag_flag) # mark. moved jax.lax.scan into update_networks
            

            return (
                training_state,
                env_state,
                buffer_state,
            ), metrics

        @jax.jit
        def training_epoch(
            protag_training_state, # mark passing in the antag training state
            antag_training_state,
            env_state,
            buffer_state,
            key,
        ):
            
            
            print("Training epoch: entered") # MARKLOG
            #@jax.jit mark
            def f(carry, unused_t):
                ts, es, bs, k = carry
                k, train_key = jax.random.split(k, 2)
                (
                    (
                        ts,
                        es,
                        bs,
                    ),
                    metrics,
                ) = training_step(protag_actor_state, antag_actor_state, protag_action_size, False, ts, es, bs, train_key) # mark added arguments
                return (ts, es, bs, k), metrics

            def g(carry, unused_t): # antag version of f
                ts, es, bs, k = carry
                k, train_key = jax.random.split(k, 2)
                (
                    (
                        ts,
                        es,
                        bs,
                    ),
                    antag_metrics,
                ) = training_step(protag_actor_state, antag_actor_state, antag_action_size, True, ts, es, bs, train_key) # mark added arguments (true for negative rewards on antagonist gradient updates)
                return (ts, es, bs, k), antag_metrics

            (protag_training_state, env_state, buffer_state, key), metrics = jax.lax.scan(
                f,
                (protag_training_state, env_state, buffer_state, key),
                (),
                length=num_training_steps_per_epoch,
            )

            (antag_training_state, env_state, buffer_state, key), antag_metrics = jax.lax.scan( # mark added antag scan training
                g,
                (antag_training_state, env_state, buffer_state, key),
                (),
                length=num_training_steps_per_epoch,
            )

            metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
            antag_metrics["buffer_current_size"] = replay_buffer.size(buffer_state) # mark antag metrics, although they use the same buffer
            return protag_training_state, antag_training_state, env_state, buffer_state, metrics, antag_metrics

        key, prefill_key = jax.random.split(key, 2)

        protag_training_state, antag_training_state, env_state, buffer_state, _ = prefill_replay_buffer( # mark adjusted return values and arguments
            protag_training_state, antag_training_state, env_state, buffer_state, prefill_key
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
            print(f"Entered training loop at eval {ne}") #MARKLOG
            logging.info(f"Entered training loop at eval {ne}") #MARKLOG
            
            t = time.time()

            key, epoch_key = jax.random.split(key)

            protag_training_state, antag_training_state, env_state, buffer_state, metrics, antag_metrics = training_epoch( # mark
                protag_training_state, antag_training_state, env_state, buffer_state, epoch_key
            )

            print("After training epoch") # MARKLOG

            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            metrics = jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

            antag_metrics = jax.tree_util.tree_map(jnp.mean, antag_metrics) # mark antag metrics
            antag_metrics = jax.tree_util.tree_map(lambda x: x.block_until_ready(), antag_metrics)

            epoch_training_time = time.time() - t
            training_walltime += epoch_training_time

            sps = (env_steps_per_actor_step * num_training_steps_per_epoch) / epoch_training_time
            metrics = {
                "training/sps": sps,
                "training/walltime": training_walltime,
                "training/envsteps": protag_training_state.env_steps.item(), # mark
                **{f"training/{name}": value for name, value in metrics.items()},
            }
            current_step = int(protag_training_state.env_steps.item())

            metrics = evaluator.run_evaluation(protag_training_state, metrics) # mark
            print("After evaluator") # MARKLOG
            logging.info("step: %d", current_step)

            do_render = ne % config.visualization_interval == 0
            make_policy = lambda param: lambda obs, rng: protag_actor.apply(param, obs) # mark

            progress_fn(
                current_step,
                metrics,
                make_policy,
                protag_training_state.actor_state.params,
                unwrapped_env,
                do_render=do_render,
            )

            print("After progress_fn") #MARKLOG

            if config.checkpoint_logdir:
                # Save current policy and critic params.
                params = (
                    protag_training_state.alpha_state.params, # mark
                    protag_training_state.actor_state.params,
                    protag_training_state.critic_state.params,
                )
                path = f"{config.checkpoint_logdir}/step_{int(protag_training_state.env_steps)}.pkl"
                save_params(path, params)

        total_steps = current_step
        assert total_steps >= config.total_env_steps

        logging.info("total steps: %s", total_steps)

        return make_policy, params, metrics
