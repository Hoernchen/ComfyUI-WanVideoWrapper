"""
Flow Matching DPM++ 2S Ancestral Scheduler - Implementation
"""
import torch


class FlowDPMPP2SAncestralScheduler:
    """
    DPM++ 2S Ancestral scheduler with two-stage evaluation.
    """

    def __init__(self, num_train_timesteps=1000, shift=3.0, eta=1.0, s_noise=1.0):
        """
        Initialize the DPM++ 2S Ancestral scheduler.

        Args:
            num_train_timesteps: Number of training timesteps
            shift: Shift parameter for flow matching
            eta: Ancestral sampling strength (0 = deterministic, 1 = full ancestral)
            s_noise: Noise scaling factor
        """
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.eta = eta
        self.s_noise = s_noise
        self._sigmas = None
        self._timesteps = None
        self.num_inference_steps = None

    def set_timesteps(self, num_inference_steps, device=None, shift=None,
                      use_beta_sigmas=False, sigmas=None, **kwargs):
        """
        Set the timesteps for the scheduler.

        Args:
            num_inference_steps: Number of inference steps
            device: Device to place tensors on
            shift: Shift parameter (overrides instance shift if provided)
            use_beta_sigmas: Whether to use beta distribution for sigmas
            sigmas: Use custom sigmas if provided
        """
        if shift is not None:
            self.shift = shift

        if sigmas is not None:
            # Use custom sigmas if provided
            self.sigmas = torch.tensor(sigmas, device=device)
            self.timesteps = (self.sigmas * 1000).to(torch.int64).to(device)
            self.num_inference_steps = len(self.sigmas)
        else:
            # Generate sigmas based on the specified distribution
            if use_beta_sigmas:
                # Beta distribution for sigmas
                betas = torch.linspace(0.00085, 0.012, num_inference_steps + 1, device=device)
                alphas = 1.0 - betas
                alphas_cumprod = torch.cumprod(alphas, dim=0)
                sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
            else:
                # Linear spacing in sigma space
                sigma_max = 1.0
                sigma_min = 0.003 / 1.002
                sigmas = torch.linspace(sigma_max, sigma_min, num_inference_steps + 1, device=device)

            # Apply shift transformation
            sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)

            # Include zero at the end for final step
            sigmas[-1] = 0.0

            self.sigmas = sigmas
            self.timesteps = (sigmas[:-1] * self.num_train_timesteps).to(torch.int64)
            self.num_inference_steps = num_inference_steps

    @property
    def sigmas(self):
        return self._sigmas

    @sigmas.setter
    def sigmas(self, value):
        self._sigmas = value

    @property
    def timesteps(self):
        return self._timesteps

    @timesteps.setter
    def timesteps(self, value):
        self._timesteps = value

    def step(self, model_output, timestep, sample, model=None, extra_args=None,
             generator=None, return_dict=True, **kwargs):
        """
        Perform one step of DPM++ 2S Ancestral with TWO model evaluations.

        Args:
            model_output: First model evaluation (can be used or re-evaluated)
            timestep: Current timestep
            sample: Current sample
            model: REQUIRED for second evaluation - The model function
            extra_args: Extra arguments for model evaluations
            generator: Random number generator for ancestral sampling
            return_dict: Whether to return a dict

        Returns:
            prev_sample: Sample at the previous timestep
        """
        if extra_args is None:
            extra_args = {}

        # Handle batched timesteps
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)

        # Ensure tensors are on the same device
        self.sigmas = self.sigmas.to(sample.device)
        self.timesteps = self.timesteps.to(sample.device)

        # Find the index of the current timestep
        if timestep.ndim == 0:
            timestep_id = torch.argmin((self.timesteps - timestep).abs(), dim=0)
        else:
            timestep_id = torch.argmin((self.timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)

        # Get current and next sigma
        sigma = self.sigmas[timestep_id]
        sigma_next = self.sigmas[timestep_id + 1]

        # Reshape sigmas for broadcasting
        if sigma.ndim > 0:
            sigma = sigma.reshape(-1, 1, 1, 1)
            sigma_next = sigma_next.reshape(-1, 1, 1, 1)
        else:
            sigma = sigma.view(1, 1, 1, 1)
            sigma_next = sigma_next.view(1, 1, 1, 1)

        # FIRST MODEL EVALUATION
        if model is not None:
            # Re-evaluate at current point to ensure consistency
            timestep_1 = sigma * 1000
            model_output = model(sample, timestep_1, **extra_args)

        # Convert flow prediction to denoised prediction
        # In flow matching: x_0 = x_t - sigma_t * v_t
        denoised = sample - sigma * model_output

        # DPM++ 2S ancestral step
        if sigma_next > 0:
            # Second-order predictor
            # Calculate midpoint sigma
            sigma_mid = torch.sqrt(sigma * sigma_next)
            dt = sigma_mid - sigma

            # Compute sample at midpoint
            sample_mid = sample - model_output * dt


            # SECOND MODEL EVALUATION (at midpoint)
            if model is not None:
                timestep_mid = sigma_mid * 1000
                model_output_mid = model(sample_mid, timestep_mid, **extra_args)
            else:
                assert(False, "wtf")

            # Second-order corrector using midpoint evaluation
            denoised_mid = sample_mid - sigma_mid * model_output_mid

            # Ancestral sampling step
            # Calculate sigma_up and sigma_down for noise injection
            if self.eta > 0:
                # Compute noise levels for ancestral sampling
                sigma_up = self.eta * torch.sqrt((sigma_next**2 / sigma**2) *
                                                 (1 - (sigma_next / sigma)**2))
                sigma_down = torch.sqrt(sigma_next**2 - sigma_up**2)
            else:
                # Deterministic (eta = 0)
                sigma_up = torch.zeros_like(sigma)
                sigma_down = sigma_next

            # Generate noise for ancestral sampling
            if sigma_up > 0 and generator is not None:
                # Generate noise with proper device handling
                if hasattr(generator, 'device'):
                    noise_device = generator.device
                else:
                    noise_device = torch.device("cpu")

                noise = torch.randn(sample.shape, generator=generator,
                                  device=noise_device, dtype=torch.float32)
                noise = noise.to(sample.device).to(sample.dtype)
                noise = noise * self.s_noise
            else:
                noise = torch.zeros_like(sample)

            # Final update using second-order formula with ancestral noise
            # This combines the predictor-corrector with ancestral sampling
            if sigma > 0:
                # Standard DPM++ 2S ancestral update
                prev_sample = (denoised_mid +
                             sigma_down * (sample - denoised) / sigma +
                             sigma_up * noise)
            else:
                # Avoid division by zero
                prev_sample = denoised_mid + sigma_up * noise
        else:
            # Last step - no noise, just use denoised
            prev_sample = denoised

        if not return_dict:
            return (prev_sample,)

        return prev_sample

    def add_noise(self, original_samples, noise, timestep):
        """
        Add noise to samples for training or initialization.

        Args:
            original_samples: Clean samples
            noise: Random noise
            timestep: Timestep for noise level

        Returns:
            Noisy samples
        """
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)

        self.sigmas = self.sigmas.to(noise.device)
        self.timesteps = self.timesteps.to(noise.device)

        timestep_id = torch.argmin(
            (self.timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma = self.sigmas[timestep_id].reshape(-1, 1, 1, 1)

        # Flow matching forward process: x_t = (1 - sigma) * x_0 + sigma * noise
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample.type_as(noise)