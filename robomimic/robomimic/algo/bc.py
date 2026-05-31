"""
Implementation of Behavioral Cloning (BC).
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

import robomimic.models.base_nets as BaseNets
import robomimic.models.obs_nets as ObsNets
import robomimic.models.policy_nets as PolicyNets
import robomimic.models.vae_nets as VAENets
import robomimic.utils.loss_utils as LossUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import torchvision.transforms as T

from robomimic.algo import register_algo_factory_func, PolicyAlgo


@register_algo_factory_func("bc")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the BC algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """

    # note: we need the check below because some configs import BCConfig and exclude
    # some of these options
    gaussian_enabled = ("gaussian" in algo_config and algo_config.gaussian.enabled)
    gmm_enabled = ("gmm" in algo_config and algo_config.gmm.enabled)
    vae_enabled = ("vae" in algo_config and algo_config.vae.enabled)

    # EDL
    edl_frame_stacking_multihead_enabled = ("edl" in algo_config and algo_config.edl_frame_stacking_multihead.enabled)
    edl_frame_stacking_enabled = ("edl" in algo_config and algo_config.edl_frame_stacking.enabled)
    edl_hybrid_frame_stacking_enabled = ("edl" in algo_config and algo_config.edl_hybrid_frame_stacking.enabled)
    edl_enabled = ("edl" in algo_config and algo_config.edl.enabled)

    rnn_enabled = algo_config.rnn.enabled
    # support legacy configs that do not have "transformer" item
    transformer_enabled = ("transformer" in algo_config) and algo_config.transformer.enabled

    if gaussian_enabled:
        if rnn_enabled:
            raise NotImplementedError
        elif transformer_enabled:
            raise NotImplementedError
        else:
            algo_class, algo_kwargs = BC_Gaussian, {}
    elif gmm_enabled:
        if rnn_enabled:
            algo_class, algo_kwargs = BC_RNN_GMM, {}
        elif transformer_enabled:
            algo_class, algo_kwargs = BC_Transformer_GMM, {}
        else:
            algo_class, algo_kwargs = BC_GMM, {}
    elif vae_enabled:
        if rnn_enabled:
            raise NotImplementedError
        elif transformer_enabled:
            raise NotImplementedError
        else:
            algo_class, algo_kwargs = BC_VAE, {}
    elif edl_enabled:
        if rnn_enabled or transformer_enabled:
            raise NotImplementedError("EDL not yet implemented for RNN/Transformer!")
        else:
            algo_class, algo_kwargs = BC_EDL, {}
    elif edl_frame_stacking_enabled:
        if rnn_enabled or transformer_enabled:
            raise NotImplementedError
        else:
            algo_class, algo_kwargs = BC_EDL_STACKING, {}
    elif edl_frame_stacking_multihead_enabled:
        if rnn_enabled or transformer_enabled:
            raise NotImplementedError
        else:
            algo_class, algo_kwargs = BC_EDL_STACKING_MULTIHEAD, {}
    elif edl_hybrid_frame_stacking_enabled:
        if rnn_enabled or transformer_enabled:
            raise NotImplementedError
        else:
            algo_class, algo_kwargs = BC_EDL_HYBRID_STACKING, {}
    else:
        if rnn_enabled:
            algo_class, algo_kwargs = BC_RNN, {}
        elif transformer_enabled:
            algo_class, algo_kwargs = BC_Transformer, {}
        else:
            algo_class, algo_kwargs = BC, {}
    
    return algo_class, algo_kwargs


class BC(PolicyAlgo):
    """
    Normal BC training.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.ActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )
        self.nets = self.nets.float().to(self.device)

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training 
        """
        input_batch = dict()
        input_batch["obs"] = {k: batch["obs"][k][:, 0, :] for k in batch["obs"]}
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present
        input_batch["actions"] = batch["actions"][:, 0, :]
        # we move to device first before float conversion because image observation modalities will be uint8 -
        # this minimizes the amount of data transferred to GPU
        return TensorUtils.to_float(TensorUtils.to_device(input_batch, self.device))


    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            validate (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(BC, self).train_on_batch(batch, epoch, validate=validate)
            predictions = self._forward_training(batch)
            losses = self._compute_losses(predictions, batch)

            info["predictions"] = TensorUtils.detach(predictions)
            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                step_info = self._train_step(losses)
                info.update(step_info)

        return info

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        predictions = OrderedDict()
        actions = self.nets["policy"](obs_dict=batch["obs"], goal_dict=batch["goal_obs"])
        predictions["actions"] = actions
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """
        losses = OrderedDict()
        a_target = batch["actions"]
        actions = predictions["actions"]
        losses["l2_loss"] = nn.MSELoss()(actions, a_target)
        losses["l1_loss"] = nn.SmoothL1Loss()(actions, a_target)
        # cosine direction loss on eef delta position
        losses["cos_loss"] = LossUtils.cosine_loss(actions[..., :3], a_target[..., :3])

        action_losses = [
            self.algo_config.loss.l2_weight * losses["l2_loss"],
            self.algo_config.loss.l1_weight * losses["l1_loss"],
            self.algo_config.loss.cos_weight * losses["cos_loss"],
        ]
        action_loss = sum(action_losses)
        losses["action_loss"] = action_loss
        return losses

    def _train_step(self, losses):
        """
        Internal helper function for BC algo class. Perform backpropagation on the
        loss tensors in @losses to update networks.

        Args:
            losses (dict): dictionary of losses computed over the batch, from @_compute_losses
        """

        # gradient step
        info = OrderedDict()
        policy_grad_norms = TorchUtils.backprop_for_loss(
            net=self.nets["policy"],
            optim=self.optimizers["policy"],
            loss=losses["action_loss"],
            max_grad_norm=self.global_config.train.max_grad_norm,
        )
        info["policy_grad_norms"] = policy_grad_norms
        return info

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = super(BC, self).log_info(info)
        log["Loss"] = info["losses"]["action_loss"].item()
        if "l2_loss" in info["losses"]:
            log["L2_Loss"] = info["losses"]["l2_loss"].item()
        if "l1_loss" in info["losses"]:
            log["L1_Loss"] = info["losses"]["l1_loss"].item()
        if "cos_loss" in info["losses"]:
            log["Cosine_Loss"] = info["losses"]["cos_loss"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor
        """
        assert not self.nets.training
        return self.nets["policy"](obs_dict, goal_dict=goal_dict)


class BC_Gaussian(BC):
    """
    BC training with a Gaussian policy.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert self.algo_config.gaussian.enabled

        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.GaussianActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            fixed_std=self.algo_config.gaussian.fixed_std,
            init_std=self.algo_config.gaussian.init_std,
            std_limits=(self.algo_config.gaussian.min_std, 7.5),
            std_activation=self.algo_config.gaussian.std_activation,
            low_noise_eval=self.algo_config.gaussian.low_noise_eval,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )

        self.nets = self.nets.float().to(self.device)

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        dists = self.nets["policy"].forward_train(
            obs_dict=batch["obs"], 
            goal_dict=batch["goal_obs"],
        )

        # make sure that this is a batch of multivariate action distributions, so that
        # the log probability computation will be correct
        assert len(dists.batch_shape) == 1
        log_probs = dists.log_prob(batch["actions"])

        predictions = OrderedDict(
            log_probs=log_probs,
        )
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """

        # loss is just negative log-likelihood of action targets
        action_loss = -predictions["log_probs"].mean()
        return OrderedDict(
            log_probs=-action_loss,
            action_loss=action_loss,
        )

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["Log_Likelihood"] = info["losses"]["log_probs"].item() 
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log


class BC_GMM(BC_Gaussian):
    """
    BC training with a Gaussian Mixture Model policy.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert self.algo_config.gmm.enabled

        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.GMMActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            num_modes=self.algo_config.gmm.num_modes,
            min_std=self.algo_config.gmm.min_std,
            std_activation=self.algo_config.gmm.std_activation,
            low_noise_eval=self.algo_config.gmm.low_noise_eval,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )

        self.nets = self.nets.float().to(self.device)


class BC_VAE(BC):
    """
    BC training with a VAE policy.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.VAEActor(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            device=self.device,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **VAENets.vae_args_from_config(self.algo_config.vae),
        )
        
        self.nets = self.nets.float().to(self.device)

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Update from superclass to set categorical temperature, for categorical VAEs.
        """
        if self.algo_config.vae.prior.use_categorical:
            temperature = self.algo_config.vae.prior.categorical_init_temp - epoch * self.algo_config.vae.prior.categorical_temp_anneal_step
            temperature = max(temperature, self.algo_config.vae.prior.categorical_min_temp)
            self.nets["policy"].set_gumbel_temperature(temperature)
        return super(BC_VAE, self).train_on_batch(batch, epoch, validate=validate)

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        vae_inputs = dict(
            actions=batch["actions"],
            obs_dict=batch["obs"],
            goal_dict=batch["goal_obs"],
            freeze_encoder=batch.get("freeze_encoder", False),
        )

        vae_outputs = self.nets["policy"].forward_train(**vae_inputs)
        predictions = OrderedDict(
            actions=vae_outputs["decoder_outputs"],
            kl_loss=vae_outputs["kl_loss"],
            reconstruction_loss=vae_outputs["reconstruction_loss"],
            encoder_z=vae_outputs["encoder_z"],
        )
        if not self.algo_config.vae.prior.use_categorical:
            with torch.no_grad():
                encoder_variance = torch.exp(vae_outputs["encoder_params"]["logvar"])
            predictions["encoder_variance"] = encoder_variance
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """

        # total loss is sum of reconstruction and KL, weighted by beta
        kl_loss = predictions["kl_loss"]
        recons_loss = predictions["reconstruction_loss"]
        action_loss = recons_loss + self.algo_config.vae.kl_weight * kl_loss
        return OrderedDict(
            recons_loss=recons_loss,
            kl_loss=kl_loss,
            action_loss=action_loss,
        )

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["KL_Loss"] = info["losses"]["kl_loss"].item()
        log["Reconstruction_Loss"] = info["losses"]["recons_loss"].item()
        if self.algo_config.vae.prior.use_categorical:
            log["Gumbel_Temperature"] = self.nets["policy"].get_gumbel_temperature()
        else:
            log["Encoder_Variance"] = info["predictions"]["encoder_variance"].mean().item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log


class BC_RNN(BC):
    """
    BC training with an RNN policy.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.RNNActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **BaseNets.rnn_args_from_config(self.algo_config.rnn),
        )

        self._rnn_hidden_state = None
        self._rnn_horizon = self.algo_config.rnn.horizon
        self._rnn_counter = 0
        self._rnn_is_open_loop = self.algo_config.rnn.get("open_loop", False)

        self.nets = self.nets.float().to(self.device)

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """
        input_batch = dict()
        input_batch["obs"] = batch["obs"]
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present
        input_batch["actions"] = batch["actions"]

        if self._rnn_is_open_loop:
            # replace the observation sequence with one that only consists of the first observation.
            # This way, all actions are predicted "open-loop" after the first observation, based
            # on the rnn hidden state.
            n_steps = batch["actions"].shape[1]
            obs_seq_start = TensorUtils.index_at_time(batch["obs"], ind=0)
            input_batch["obs"] = TensorUtils.unsqueeze_expand_at(obs_seq_start, size=n_steps, dim=1)

        # we move to device first before float conversion because image observation modalities will be uint8 -
        # this minimizes the amount of data transferred to GPU
        return TensorUtils.to_float(TensorUtils.to_device(input_batch, self.device))

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor
        """
        assert not self.nets.training

        if self._rnn_hidden_state is None or self._rnn_counter % self._rnn_horizon == 0:
            batch_size = list(obs_dict.values())[0].shape[0]
            self._rnn_hidden_state = self.nets["policy"].get_rnn_init_state(batch_size=batch_size, device=self.device)

            if self._rnn_is_open_loop:
                # remember the initial observation, and use it instead of the current observation
                # for open-loop action sequence prediction
                self._open_loop_obs = TensorUtils.clone(TensorUtils.detach(obs_dict))

        obs_to_use = obs_dict
        if self._rnn_is_open_loop:
            # replace current obs with last recorded obs
            obs_to_use = self._open_loop_obs

        self._rnn_counter += 1
        action, self._rnn_hidden_state = self.nets["policy"].forward_step(
            obs_to_use, goal_dict=goal_dict, rnn_state=self._rnn_hidden_state)
        return action

    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        self._rnn_hidden_state = None
        self._rnn_counter = 0


class BC_RNN_GMM(BC_RNN):
    """
    BC training with an RNN GMM policy.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert self.algo_config.gmm.enabled
        assert self.algo_config.rnn.enabled

        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.RNNGMMActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            num_modes=self.algo_config.gmm.num_modes,
            min_std=self.algo_config.gmm.min_std,
            std_activation=self.algo_config.gmm.std_activation,
            low_noise_eval=self.algo_config.gmm.low_noise_eval,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **BaseNets.rnn_args_from_config(self.algo_config.rnn),
        )

        self._rnn_hidden_state = None
        self._rnn_horizon = self.algo_config.rnn.horizon
        self._rnn_counter = 0
        self._rnn_is_open_loop = self.algo_config.rnn.get("open_loop", False)

        self.nets = self.nets.float().to(self.device)

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        dists = self.nets["policy"].forward_train(
            obs_dict=batch["obs"], 
            goal_dict=batch["goal_obs"],
        )

        # make sure that this is a batch of multivariate action distributions, so that
        # the log probability computation will be correct
        assert len(dists.batch_shape) == 2 # [B, T]
        log_probs = dists.log_prob(batch["actions"])

        predictions = OrderedDict(
            log_probs=log_probs,
        )
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """

        # loss is just negative log-likelihood of action targets
        action_loss = -predictions["log_probs"].mean()
        return OrderedDict(
            log_probs=-action_loss,
            action_loss=action_loss,
        )

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["Log_Likelihood"] = info["losses"]["log_probs"].item() 
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log


class BC_Transformer(BC):
    """
    BC training with a Transformer policy.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert self.algo_config.transformer.enabled

        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.TransformerActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **BaseNets.transformer_args_from_config(self.algo_config.transformer),
        )
        self._set_params_from_config()
        self.nets = self.nets.float().to(self.device)
        
    def _set_params_from_config(self):
        """
        Read specific config variables we need for training / eval.
        Called by @_create_networks method
        """
        self.context_length = self.algo_config.transformer.context_length
        self.supervise_all_steps = self.algo_config.transformer.supervise_all_steps
        self.pred_future_acs = self.algo_config.transformer.pred_future_acs
        if self.pred_future_acs:
            assert self.supervise_all_steps is True

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader
        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """
        input_batch = dict()
        h = self.context_length
        input_batch["obs"] = {k: batch["obs"][k][:, :h, :] for k in batch["obs"]}
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present

        if self.supervise_all_steps:
            # supervision on entire sequence (instead of just current timestep)
            if self.pred_future_acs:
                ac_start = h - 1
            else:
                ac_start = 0
            input_batch["actions"] = batch["actions"][:, ac_start:ac_start+h, :]
        else:
            # just use current timestep
            input_batch["actions"] = batch["actions"][:, h-1, :]

        if self.pred_future_acs:
            assert input_batch["actions"].shape[1] == h

        input_batch = TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)
        return input_batch

    def _forward_training(self, batch, epoch=None):
        """
        Internal helper function for BC_Transformer algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        # ensure that transformer context length is consistent with temporal dimension of observations
        TensorUtils.assert_size_at_dim(
            batch["obs"], 
            size=(self.context_length), 
            dim=1, 
            msg="Error: expect temporal dimension of obs batch to match transformer context length {}".format(self.context_length),
        )

        predictions = OrderedDict()
        predictions["actions"] = self.nets["policy"](obs_dict=batch["obs"], actions=None, goal_dict=batch["goal_obs"])
        if not self.supervise_all_steps:
            # only supervise final timestep
            predictions["actions"] = predictions["actions"][:, -1, :]
        return predictions

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.
        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal
        Returns:
            action (torch.Tensor): action tensor
        """
        assert not self.nets.training

        output = self.nets["policy"](obs_dict, actions=None, goal_dict=goal_dict)

        if self.supervise_all_steps:
            if self.algo_config.transformer.pred_future_acs:
                output = output[:, 0, :]
            else:
                output = output[:, -1, :]
        else:
            output = output[:, -1, :]

        return output

        

class BC_Transformer_GMM(BC_Transformer):
    """
    BC training with a Transformer GMM policy.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert self.algo_config.gmm.enabled
        assert self.algo_config.transformer.enabled

        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.TransformerGMMActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            num_modes=self.algo_config.gmm.num_modes,
            min_std=self.algo_config.gmm.min_std,
            std_activation=self.algo_config.gmm.std_activation,
            low_noise_eval=self.algo_config.gmm.low_noise_eval,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **BaseNets.transformer_args_from_config(self.algo_config.transformer),
        )
        self._set_params_from_config()
        self.nets = self.nets.float().to(self.device)

    def _forward_training(self, batch, epoch=None):
        """
        Modify from super class to support GMM training.
        """
        # ensure that transformer context length is consistent with temporal dimension of observations
        TensorUtils.assert_size_at_dim(
            batch["obs"], 
            size=(self.context_length), 
            dim=1, 
            msg="Error: expect temporal dimension of obs batch to match transformer context length {}".format(self.context_length),
        )

        dists = self.nets["policy"].forward_train(
            obs_dict=batch["obs"], 
            actions=None,
            goal_dict=batch["goal_obs"],
            low_noise_eval=False,
        )

        # make sure that this is a batch of multivariate action distributions, so that
        # the log probability computation will be correct
        assert len(dists.batch_shape) == 2 # [B, T]

        if not self.supervise_all_steps:
            # only use final timestep prediction by making a new distribution with only final timestep.
            # This essentially does `dists = dists[:, -1]`
            component_distribution = D.Normal(
                loc=dists.component_distribution.base_dist.loc[:, -1],
                scale=dists.component_distribution.base_dist.scale[:, -1],
            )
            component_distribution = D.Independent(component_distribution, 1)
            mixture_distribution = D.Categorical(logits=dists.mixture_distribution.logits[:, -1])
            dists = D.MixtureSameFamily(
                mixture_distribution=mixture_distribution,
                component_distribution=component_distribution,
            )

        log_probs = dists.log_prob(batch["actions"])

        predictions = OrderedDict(
            log_probs=log_probs,
        )
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC_Transformer_GMM algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.
        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training
        Returns:
            losses (dict): dictionary of losses computed over the batch
        """

        # loss is just negative log-likelihood of action targets
        action_loss = -predictions["log_probs"].mean()
        return OrderedDict(
            log_probs=-action_loss,
            action_loss=action_loss,
        )

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.
        Args:
            info (dict): dictionary of info
        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["Log_Likelihood"] = info["losses"]["log_probs"].item() 
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log


class BC_EDL(BC):
    """
    BC training with Evidential Deep Learning (EDL).
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """

        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.EDLActorNetwork(
            obs_shapes=self.obs_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            goal_shapes=self.goal_shapes,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )

        self.nets = self.nets.float().to(self.device)

    def _forward_training(self, batch):
        """
        Passes the batch through the network to get the 4 EDL parameters.
        """
        # Returns a dict: {"gamma": ..., "v": ..., "alpha": ..., "beta": ...}
        nig_params = self.nets["policy"].forward_train(
            obs_dict=batch["obs"], 
            goal_dict=batch["goal_obs"],
        )
        
        # Pass the whole dictionary forward to the loss function
        return nig_params
    
    def train_on_batch(self, batch, epoch, validate=False):
        self.current_epoch = epoch
        return super().train_on_batch(batch, epoch, validate)

    def _compute_losses(self, predictions, batch):
        """
        Computes the Evidential Loss (NIG NLL + Regularization) with strict mathematical safeguards.
        """
        target = batch["actions"]
        gamma = predictions["gamma"]
        
        # v must be > 0 to prevent torch.log(inf) and division by zero.
        v = torch.clamp(predictions["v"], min=1e-4)
        
        # alpha must be strictly > 1.0 for the uncertainty fractions to exist.
        alpha = torch.clamp(predictions["alpha"], min=1.0001) 
        
        # beta must be > 0 to prevent torch.log(0) in the NLL.
        beta = torch.clamp(predictions["beta"], min=1e-4)

        # Negative Log-Likelihood (NLL) of the Student-t distribution
        twoB = 2 * beta
        nll = (
            0.5 * torch.log(torch.pi / v)
            - alpha * torch.log(twoB)
            + (alpha + 0.5) * torch.log(v * (target - gamma) ** 2 + twoB)
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + 0.5)
        )

        # Reg
        error = torch.abs(target - gamma)
        reg = error * (2 * v + alpha)

        total_epochs = self.global_config.train.num_epochs

        # Get config variables
        max_coeff = self.algo_config.edl.regularizer_coeff 
        warmup_percentage = self.algo_config.edl.warmup_percentage

        # Calculate how many epochs the warmup should last
        warmup_epochs = total_epochs * warmup_percentage

        # Use the intercepted current epoch to calculate the ratio
        current_ep = getattr(self, "current_epoch", 0)
        
        if warmup_epochs > 0:
            anneal_ratio = min(1.0, current_ep / warmup_epochs)
        else:
            anneal_ratio = 1.0

        current_coeff = max_coeff * anneal_ratio

        action_loss = (nll + current_coeff * reg).mean()

        # Calculate uncertainties for logging
        with torch.no_grad():
            epistemic = (beta / (v * (alpha - 1.0))).mean()
            aleatoric = (beta / (alpha - 1.0)).mean()

        return OrderedDict(
            action_loss=action_loss,
            nll_loss=nll.mean(),
            reg_loss=reg.mean(),
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric
        )

    def log_info(self, info):
        """
        Process info dictionary to pass to tensorboard.
        """
        log = PolicyAlgo.log_info(self, info)
        
        log["Loss"] = info["losses"]["action_loss"].item()
        log["NLL_Loss"] = info["losses"]["nll_loss"].item()
        log["Reg_Loss"] = info["losses"]["reg_loss"].item()
        log["Epistemic_Uncertainty"] = info["losses"]["epistemic_uncertainty"].item()
        log["Aleatoric_Uncertainty"] = info["losses"]["aleatoric_uncertainty"].item()
        
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
            
        return log
    

class BC_EDL_STACKING(BC):
    """
    BC training with Evidential Deep Learning (EDL).
    Now with rigorous Temporal Channel Stacking for perceptual aliasing.
    """
    def _create_networks(self):
        """
        Tricks the visual encoder into expecting stacked channels (e.g., 9 instead of 3)
        and stacked low-dim inputs to process sequence memory.
        """
        self.seq_len = self.global_config.train.seq_length 

        stacked_obs_shapes = OrderedDict()
        for k, v in self.obs_shapes.items():
            if len(v) == 3: 
                stacked_obs_shapes[k] = [v[0] * self.seq_len, 84, 84]
            elif len(v) == 1: 
                stacked_obs_shapes[k] = [v[0] * self.seq_len]
            else:
                stacked_obs_shapes[k] = v

        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.EDLActorNetwork(
            obs_shapes=stacked_obs_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            goal_shapes=self.goal_shapes,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )
        self.nets = self.nets.float().to(self.device)

    def _stack_temporal_obs(self, obs_dict):
        if not obs_dict:
            return None
            
        stacked = OrderedDict()
        for k, v in obs_dict.items():
            if isinstance(v, torch.Tensor) and v.dim() == 5:
                B, T, C, H, W = v.shape
                stacked[k] = v.reshape(B, T * C, H, W)
            elif isinstance(v, torch.Tensor) and v.dim() == 3:
                B, T, D = v.shape
                stacked[k] = v.reshape(B, T * D)
            else:
                stacked[k] = v
        return stacked

    def _forward_training(self, batch):
        stacked_obs = self._stack_temporal_obs(batch["obs"])
        
        raw_goal = batch.get("goal_obs", None)
        stacked_goal = self._stack_temporal_obs(raw_goal) if raw_goal is not None else None
        
        return self.nets["policy"].forward_train(obs_dict=stacked_obs, goal_dict=stacked_goal)
    
    def process_batch_for_training(self, batch):
        """
        Overrides the base BC class to PREVENT it from silently deleting 
        the temporal sequence before it reaches our stacking logic.
        """
        input_batch = dict()
        
        # Keep the full [B, T, ...] sequence for all observations
        input_batch["obs"] = {k: batch["obs"][k] for k in self.obs_shapes.keys()}
        input_batch["goal_obs"] = batch.get("goal_obs", None)
        
        # Keep the full sequence of actions 
        input_batch["actions"] = batch["actions"]

        # Ensure everything is on the GPU and formatted as floats
        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)

    def train_on_batch(self, batch, epoch, validate=False):
        self.current_epoch = epoch
        return super().train_on_batch(batch, epoch, validate)

    def _compute_losses(self, predictions, batch):
        target = batch["actions"]
        
        # When using seq_length=3, actions are shape [B, T, Action_Dim]
        # We only want to predict the action for the most recent frame
        if target.dim() == 3:
            target = target[:, -1, :]

        gamma = predictions["gamma"]
        v = torch.clamp(predictions["v"], min=1e-4)
        alpha = torch.clamp(predictions["alpha"], min=1.0001, max=10.0) 
        beta = torch.clamp(predictions["beta"], min=1e-4)

        twoB = 2 * beta
        nll = (
            0.5 * torch.log(torch.pi / v)
            - alpha * torch.log(twoB)
            + (alpha + 0.5) * torch.log(v * (target - gamma) ** 2 + twoB)
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + 0.5)
        )

        error = torch.abs(target - gamma)
        reg = error * (2 * v + alpha)
        
        total_epochs = self.global_config.train.num_epochs
        max_coeff = self.algo_config.edl_frame_stacking.regularizer_coeff 
        warmup_percentage = self.algo_config.edl_frame_stacking.warmup_percentage

        warmup_epochs = total_epochs * warmup_percentage
        current_ep = getattr(self, "current_epoch", 0)
        
        if warmup_epochs > 0:
            anneal_ratio = min(1.0, current_ep / warmup_epochs)
        else:
            anneal_ratio = 1.0

        current_coeff = max_coeff * anneal_ratio
        action_loss = (nll + current_coeff * reg).mean()

        with torch.no_grad():
            epistemic = (beta / (v * (alpha - 1.0))).mean()
            aleatoric = (beta / (alpha - 1.0)).mean()

            mean_gamma = gamma.mean()
            mean_v = v.mean()
            mean_alpha = alpha.mean()
            mean_beta = beta.mean()

        return OrderedDict(
            action_loss=action_loss,
            nll_loss=nll.mean(),
            reg_loss=reg.mean(),
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            current_coeff=torch.tensor(current_coeff),
            mean_gamma=mean_gamma,  
            mean_v=mean_v,          
            mean_alpha=mean_alpha,  
            mean_beta=mean_beta  
        )
    
    def log_info(self, info):
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["NLL_Loss"] = info["losses"]["nll_loss"].item()
        log["Reg_Loss"] = info["losses"]["reg_loss"].item()
        log["Epistemic_Uncertainty"] = info["losses"]["epistemic_uncertainty"].item()
        log["Aleatoric_Uncertainty"] = info["losses"]["aleatoric_uncertainty"].item()
        log["EDL_Regularizer_Weight"] = info["losses"]["current_coeff"].item()

        log["Evidence/Gamma_Mean"] = info["losses"]["mean_gamma"].item()
        log["Evidence/V_Mean"] = info["losses"]["mean_v"].item()
        log["Evidence/Alpha_Mean"] = info["losses"]["mean_alpha"].item()
        log["Evidence/Beta_Mean"] = info["losses"]["mean_beta"].item()

        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log


class BC_EDL_HYBRID_STACKING(BC):
    """
    BC training with Evidential Deep Learning (EDL).
    Now with rigorous Temporal Channel Stacking for perceptual aliasing
    and Hybrid UR-ER + NSUR loss to prevent gradient death and bound evidence.
    """
    def _create_networks(self):
        """
        Tricks the visual encoder into expecting stacked channels (e.g., 9 instead of 3)
        and stacked low-dim inputs to process sequence memory.
        """
        self.seq_len = self.global_config.train.seq_length  # Our temporal memory size

        stacked_obs_shapes = OrderedDict()
        for k, v in self.obs_shapes.items():
            if len(v) == 3:
                stacked_obs_shapes[k] = [v[0] * self.seq_len, v[1], v[2]] 
            elif len(v) == 1:
                stacked_obs_shapes[k] = [v[0] * self.seq_len]
            else:
                stacked_obs_shapes[k] = v

        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.EDLActorNetwork(
            obs_shapes=stacked_obs_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            goal_shapes=self.goal_shapes,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )
        self.nets = self.nets.float().to(self.device)

    def _stack_temporal_obs(self, obs_dict):
        if not obs_dict:
            return None
            
        stacked = OrderedDict()
        for k, v in obs_dict.items():
            if isinstance(v, torch.Tensor) and v.dim() == 5:
                B, T, C, H, W = v.shape
                stacked[k] = v.reshape(B, T * C, H, W)
            elif isinstance(v, torch.Tensor) and v.dim() == 3:
                B, T, D = v.shape
                stacked[k] = v.reshape(B, T * D)
            else:
                stacked[k] = v
        return stacked

    def _forward_training(self, batch):
        stacked_obs = self._stack_temporal_obs(batch["obs"])
        
        raw_goal = batch.get("goal_obs", None)
        stacked_goal = self._stack_temporal_obs(raw_goal) if raw_goal is not None else None
        
        return self.nets["policy"].forward_train(obs_dict=stacked_obs, goal_dict=stacked_goal)
    
    def process_batch_for_training(self, batch):
        """
        Overrides the base BC class to PREVENT it from silently deleting 
        the temporal sequence before it reaches our stacking logic.
        """
        input_batch = dict()
        
        # Keep the full [B, T, ...] sequence for all observations
        input_batch["obs"] = {k: batch["obs"][k] for k in self.obs_shapes.keys()}
        input_batch["goal_obs"] = batch.get("goal_obs", None)
        
        # Keep the full sequence of actions 
        input_batch["actions"] = batch["actions"]

        # Ensure everything is on the GPU and formatted as floats
        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)

    def train_on_batch(self, batch, epoch, validate=False):
        self.current_epoch = epoch
        return super().train_on_batch(batch, epoch, validate)

    def _compute_losses(self, predictions, batch):
        target = batch["actions"]
        
        # When using seq_length=3, actions are shape [B, T, Action_Dim]
        # We only want to predict the action for the most recent frame
        if target.dim() == 3:
            target = target[:, -1, :]

        # Extract parameters (v corresponds to lambda_ in the literature)
        gamma = predictions["gamma"]
        v = torch.clamp(predictions["v"], min=1e-4)
        alpha = torch.clamp(predictions["alpha"], min=1.0001, max=10.0)
        beta = torch.clamp(predictions["beta"], min=1e-4)
        
        eps = 1e-8

        twoBv = 2 * beta * (1 + v)
        nll = (
            0.5 * torch.log(torch.pi / v)
            - alpha * torch.log(twoBv)
            + (alpha + 0.5) * torch.log(v * (target - gamma) ** 2 + twoBv)
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + 0.5)
        )

        nll_mean = nll.mean()
        # Prevents gradients from dying for predictive mean
        error_abs = torch.abs(target - gamma)
        x = alpha - 1.0
        
        # Log-space factorization for numerical stability (from before)
        stable_penalty = x + torch.log(-torch.expm1(-x) + eps)
        
        ur_penalty = torch.where(
            alpha < 2.0, 
            stable_penalty, 
            torch.tensor(0.0, device=alpha.device)
        )
        
        # The negative sign now safely pushes alpha UP, but stops at 2.0
        reg_ur = -error_abs * ur_penalty
        reg_ur_mean = reg_ur.mean()

        # Absolute mathematical bound against evidence contraction
        error_sq = (target - gamma.detach()) ** 2
        inverse_total_uncertainty = (v * (alpha - 1.0)) / (beta * (v + 1.0) + eps)
        reg_nsur = 0.5 * error_sq * inverse_total_uncertainty
        reg_nsur_mean = reg_nsur.mean()

        total_epochs = self.global_config.train.num_epochs
        max_nsur_coeff = self.algo_config.edl_hybrid_frame_stacking.nsur_regularizer_coeff 
        max_ur_coeff = self.algo_config.edl_hybrid_frame_stacking.ur_regularizer_coeff 
        warmup_percentage = self.algo_config.edl_hybrid_frame_stacking.warmup_percentage

        warmup_epochs = total_epochs * warmup_percentage
        current_ep = getattr(self, "current_epoch", 0)
        
        if warmup_epochs > 0:
            anneal_ratio = min(1.0, current_ep / warmup_epochs)
        else:
            anneal_ratio = 1.0

        current_nsur_coeff = max_nsur_coeff * anneal_ratio
        current_ur_coeff = max_ur_coeff * anneal_ratio
        
        # Combine everything
        action_loss = nll_mean + (current_ur_coeff * reg_ur_mean) + (current_nsur_coeff * reg_nsur_mean)

        with torch.no_grad():
            epistemic = (beta / (v * (alpha - 1.0))).mean()
            aleatoric = (beta / (alpha - 1.0)).mean()

            # Track raw evidence to check for pinning and overall dynamics
            mean_gamma = gamma.mean()
            mean_v = v.mean()
            mean_alpha = alpha.mean()
            mean_beta = beta.mean()

        return OrderedDict(
            action_loss=action_loss,
            nll_loss=nll_mean,
            ur_reg_loss=reg_ur_mean,
            nsur_reg_loss=reg_nsur_mean,
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            current_ur_coeff=torch.tensor(current_ur_coeff),
            current_nsur_coeff=torch.tensor(current_nsur_coeff),
            mean_gamma=mean_gamma,  
            mean_v=mean_v,          
            mean_alpha=mean_alpha,  
            mean_beta=mean_beta     
        )

    def log_info(self, info):
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["NLL_Loss"] = info["losses"]["nll_loss"].item()
        log["UR_Reg_Loss"] = info["losses"]["ur_reg_loss"].item()       
        log["NSUR_Reg_Loss"] = info["losses"]["nsur_reg_loss"].item()   
        log["Epistemic_Uncertainty"] = info["losses"]["epistemic_uncertainty"].item()
        log["Aleatoric_Uncertainty"] = info["losses"]["aleatoric_uncertainty"].item()
        log["UR_Regularizer_Weight"] = info["losses"]["current_ur_coeff"].item()     
        log["NSUR_Regularizer_Weight"] = info["losses"]["current_nsur_coeff"].item() 

        log["Evidence/Gamma_Mean"] = info["losses"]["mean_gamma"].item()
        log["Evidence/V_Mean"] = info["losses"]["mean_v"].item()
        log["Evidence/Alpha_Mean"] = info["losses"]["mean_alpha"].item()
        log["Evidence/Beta_Mean"] = info["losses"]["mean_beta"].item()

        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log
    


class BC_EDL_STACKING_MULTIHEAD(BC):
    """
    BC training with Evidential Deep Learning (EDL).
    Now with rigorous Temporal Channel Stacking for perceptual aliasing.
    """
    def _create_networks(self):
        """
        Tricks the visual encoder into expecting stacked channels (e.g., 9 instead of 3)
        and stacked low-dim inputs to process sequence memory.
        """
        self.seq_len = self.global_config.train.seq_length

        stacked_obs_shapes = OrderedDict()
        for k, v in self.obs_shapes.items():
            if len(v) == 3:
                stacked_obs_shapes[k] = [v[0] * self.seq_len, 84, 84]
            elif len(v) == 1:
                stacked_obs_shapes[k] = [v[0] * self.seq_len]
            else:
                stacked_obs_shapes[k] = v

        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.EDLActorNetwork(
            obs_shapes=stacked_obs_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            goal_shapes=self.goal_shapes,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )
        self.nets = self.nets.float().to(self.device)

    def _stack_temporal_obs(self, obs_dict):
        if not obs_dict:
            return None
            
        stacked = OrderedDict()
        for k, v in obs_dict.items():
            if isinstance(v, torch.Tensor) and v.dim() == 5:
                B, T, C, H, W = v.shape
                stacked[k] = v.reshape(B, T * C, H, W)
            elif isinstance(v, torch.Tensor) and v.dim() == 3:
                B, T, D = v.shape
                stacked[k] = v.reshape(B, T * D)
            else:
                stacked[k] = v
        return stacked

    def _forward_training(self, batch):
        stacked_obs = self._stack_temporal_obs(batch["obs"])
        
        raw_goal = batch.get("goal_obs", None)
        stacked_goal = self._stack_temporal_obs(raw_goal) if raw_goal is not None else None
        
        return self.nets["policy"].forward_train(obs_dict=stacked_obs, goal_dict=stacked_goal)
    
    def process_batch_for_training(self, batch):
        """
        Overrides the base BC class to PREVENT it from silently deleting 
        the temporal sequence before it reaches our stacking logic.
        """
        input_batch = dict()
        
        # Keep the full [B, T, ...] sequence for all observations
        input_batch["obs"] = {k: batch["obs"][k] for k in self.obs_shapes.keys()}
        input_batch["goal_obs"] = batch.get("goal_obs", None)
        
        # Keep the full sequence of actions 
        input_batch["actions"] = batch["actions"]

        # Ensure everything is on the GPU and formatted as floats
        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)

    def train_on_batch(self, batch, epoch, validate=False):
        self.current_epoch = epoch
        return super().train_on_batch(batch, epoch, validate)

    def _compute_losses(self, predictions, batch):
        target = batch["actions"]
        
        # When using seq_length=3, actions are shape [B, T, Action_Dim]. 
        # We only want to predict the action for the most recent frame!
        if target.dim() == 3:
            target = target[:, -1, :]

        arm_target = target[:, :6]
        
        gripper_target = (target[:, 6] + 1.0) / 2.0 

        gamma_arm = predictions["gamma"][:, :6]
        v_arm = torch.clamp(predictions["v"][:, :6], min=1e-4)
        alpha_arm = torch.clamp(predictions["alpha"][:, :6], min=1.0001) 
        beta_arm = torch.clamp(predictions["beta"][:, :6], min=1e-4)

        twoB = 2 * beta_arm
        nll = (
            0.5 * torch.log(torch.pi / v_arm)
            - alpha_arm * torch.log(twoB)
            + (alpha_arm + 0.5) * torch.log(v_arm * (arm_target - gamma_arm) ** 2 + twoB)
            + torch.lgamma(alpha_arm)
            - torch.lgamma(alpha_arm + 0.5)
        )

        error = torch.abs(arm_target - gamma_arm)
        reg = error * (2 * v_arm + alpha_arm)

        # Hijack the raw "gamma" output of the 7th dim to use as our BCE logit
        gripper_logit = predictions["gamma"][:, 6]
        
        bce_loss_fn = nn.BCEWithLogitsLoss()
        bce_loss = bce_loss_fn(gripper_logit, gripper_target)

        gripper_weight = self.algo_config.edl_frame_stacking_multihead.gripper_coefficient 
        weighted_bce = bce_loss * gripper_weight

        total_epochs = self.global_config.train.num_epochs
        max_coeff = self.algo_config.edl_frame_stacking_multihead.regularizer_coeff 
        warmup_percentage = self.algo_config.edl_frame_stacking_multihead.warmup_percentage

        warmup_epochs = total_epochs * warmup_percentage
        current_ep = getattr(self, "current_epoch", 0)
        
        if warmup_epochs > 0:
            anneal_ratio = min(1.0, current_ep / warmup_epochs)
        else:
            anneal_ratio = 1.0

        current_coeff = max_coeff * anneal_ratio

        action_loss = (nll + current_coeff * reg).mean() + weighted_bce

        with torch.no_grad():
            epistemic = (beta_arm / (v_arm * (alpha_arm - 1.0))).mean()
            aleatoric = (beta_arm / (alpha_arm - 1.0)).mean()

        return OrderedDict(
            action_loss=action_loss,
            nll_loss=nll.mean(),
            reg_loss=reg.mean(),
            bce_loss=bce_loss,
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            current_coeff=torch.tensor(current_coeff)
        )

    def log_info(self, info):
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["NLL_Loss"] = info["losses"]["nll_loss"].item()
        log["Reg_Loss"] = info["losses"]["reg_loss"].item()
        log["Gripper_BCE_Loss"] = info["losses"]["bce_loss"].item()
        log["Epistemic_Uncertainty"] = info["losses"]["epistemic_uncertainty"].item()
        log["Aleatoric_Uncertainty"] = info["losses"]["aleatoric_uncertainty"].item()
        log["EDL_Regularizer_Weight"] = info["losses"]["current_coeff"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log