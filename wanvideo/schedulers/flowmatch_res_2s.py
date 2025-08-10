"""
Flow Matching RES 2S Scheduler
Implements the 2nd order single-step Runge-Kutta method with exponential integrators
and optional BONGMATH refinement.
"""
import torch
import numpy as np


class FlowRes2sScheduler:
    """
    RES 2S scheduler with two-stage evaluation and optional BONGMATH.
    Full RK method that requires TWO model evaluations per step.
    """

    def __init__(self, num_train_timesteps=1000, shift=3.0, c2=0.5, use_bongmath=False):
        """
        Initialize the RES 2S scheduler.

        Args:
            num_train_timesteps: Number of training timesteps
            shift: Shift parameter for flow matching
            c2: RK coefficient for intermediate point (default 0.5)
            use_bongmath: Whether to use BONGMATH iterative refinement
        """
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.c2 = c2
        self.use_bongmath = use_bongmath
        self._sigmas = None
        self._timesteps = None
        self.num_inference_steps = None

    def phi_1(self, h):
        """
        Compute phi_1(h) = (exp(h) - 1) / h for h != 0, else 1.
        This is an exponential integrator function.
        """
        if isinstance(h, torch.Tensor):
            small_h = torch.abs(h) < 1e-7
            result = torch.where(small_h,
                                torch.ones_like(h),
                                torch.expm1(h) / h)
        else:
            if abs(h) < 1e-7:
                return 1.0
            return (np.exp(h) - 1.0) / h
        return result

    def phi_2(self, h):
        """
        Compute phi_2(h) = (exp(h) - 1 - h) / h^2 for h != 0, else 1/2.
        This is an exponential integrator function.
        """
        if isinstance(h, torch.Tensor):
            small_h = torch.abs(h) < 1e-7
            result = torch.where(small_h,
                                0.5 * torch.ones_like(h),
                                (torch.expm1(h) - h) / (h * h))
        else:
            if abs(h) < 1e-7:
                return 0.5
            return (np.exp(h) - 1.0 - h) / (h * h)
        return result

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
            kwargs: wtf am I doing.
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

            # Apply shift transformation for flow matching
            sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)

            # Keep all sigmas including the last one
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

    def bongmath_iteration(self, sample, sigma, sigma_next, h, k1, k2,
                          model, extra_args, iterations=100):
        """
        BONGMATH iterative refinement.
        Enforces consistency by iterating to find fixed point between x_0,
        intermediate RK stages, and velocity predictions.

        Args:
            sample: Current sample (x at current timestep)
            sigma: Current sigma
            sigma_next: Next sigma
            h: Step size (log(sigma_next/sigma))
            k1: First stage velocity
            k2: Second stage velocity
            model: The model function for evaluations
            extra_args: Extra arguments for model
            iterations: Number of BONGMATH iterations

        Returns:
            Refined (sample, k1, k2) tuple
        """
        if not self.use_bongmath or iterations == 0:
            return sample, k1, k2

        # Store channel means if preserving them
        preserve_ch_means = True  # Could make this configurable
        if preserve_ch_means:
            if sample.ndim == 4:
                ch_mean_dims = (-2, -1)
            else:  # video tensors
                ch_mean_dims = (-4, -2, -1)
            sample_means = sample.mean(dim=ch_mean_dims, keepdim=True)

        # Phi functions for the iteration
        phi_1_h = self.phi_1(h)
        phi_2_h = self.phi_2(h)
        phi_1_c2h = self.phi_1(self.c2 * h)

        # RK weights
        b1 = phi_1_h - phi_2_h / self.c2
        b2 = phi_2_h / self.c2
        a2_1 = self.c2 * phi_1_c2h

        # BONGMATH fixed-point iteration
        for i in range(iterations):
            # Step 1: Current sample estimate
            sample_new = sample  # Start from current sample

            # Preserve channel means if requested
            if preserve_ch_means:
                sample_new = sample_new - sample_new.mean(dim=ch_mean_dims, keepdim=True) + sample_means

            # Step 2: Recompute k1 with current sample (FIRST MODEL EVALUATION IN ITERATION)
            # Note: In flow matching, the model predicts velocity v(x, sigma)
            timestep_1 = sigma * 1000
            k1_new = model(sample_new, timestep_1, **extra_args)

            # Step 3: Recompute intermediate point and k2 (SECOND MODEL EVALUATION IN ITERATION)
            sigma_mid = sigma * torch.exp(self.c2 * h)
            x_mid = sample_new + (sigma_mid - sigma) * a2_1 * k1_new

            timestep_2 = sigma_mid * 1000
            k2_new = model(x_mid, timestep_2, **extra_args)

            # Check convergence
            if (torch.allclose(k1, k1_new, rtol=1e-6) and
                torch.allclose(k2, k2_new, rtol=1e-6)):
                break

            # Update for next iteration
            k1 = k1_new
            k2 = k2_new

            # Update sample based on new k values
            sample_next = sample + (sigma_next - sigma) * (b1 * k1 + b2 * k2)
            if torch.allclose(sample, sample_next, rtol=1e-6):
                sample = sample_next
                break
            sample = sample_next

        return sample, k1, k2

    def step(self, model_output, timestep, sample, model=None, extra_args=None,
             generator=None, return_dict=True, **kwargs):
        """
        Perform one step of the RES 2S method with TWO model evaluations.

        IMPORTANT: This method REQUIRES the model parameter to be provided.
        The model_output parameter is IGNORED because we need to control both evaluations.

        Args:
            model_output: IGNORED - we perform our own evaluations
            timestep: Current discrete timestep
            sample: Current sample
            model: REQUIRED - The model function for evaluations
            extra_args: Extra arguments for model evaluations
            generator: Random number generator (not used)
            return_dict: Whether to return a dict

        Returns:
            prev_sample: Sample at the previous timestep
        """
        if model is None:
            raise ValueError("RES 2S scheduler REQUIRES the model parameter. "
                           "The scheduler must control both model evaluations.")

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
        sigma_next = self.sigmas[timestep_id + 1] if timestep_id + 1 < len(self.sigmas) else torch.zeros_like(sigma)

        # Reshape sigmas for broadcasting
        if sigma.ndim > 0:
            sigma = sigma.reshape(-1, 1, 1, 1)
            sigma_next = sigma_next.reshape(-1, 1, 1, 1)
        else:
            sigma = sigma.view(1, 1, 1, 1)
            sigma_next = sigma_next.view(1, 1, 1, 1)

        # Calculate step size h
        if sigma_next > 0:
            h = torch.log(sigma_next / sigma)
        else:
            h = torch.tensor(-10.0, device=sigma.device)  # Large negative for final step

        # Calculate phi functions (exponential integrators)
        phi_1_h = self.phi_1(h)
        phi_2_h = self.phi_2(h)
        phi_1_c2h = self.phi_1(self.c2 * h)

        # Calculate RK tableau coefficients
        # These come from the res_2s RK tableau
        a2_1 = self.c2 * phi_1_c2h  # Coefficient for intermediate point
        b1 = phi_1_h - phi_2_h / self.c2  # Weight for first stage
        b2 = phi_2_h / self.c2  # Weight for second stage


        # FIRST MODEL EVALUATION (Stage 1)
        # Evaluate at current point (sample, sigma)
        timestep_1 = sigma * 1000  # Convert sigma to timestep
        k1 = model(sample, timestep_1, **extra_args)


        # SECOND MODEL EVALUATION (Stage 2)
        # Calculate intermediate point
        sigma_mid = sigma * torch.exp(self.c2 * h)
        x_mid = sample + (sigma_mid - sigma) * a2_1 * k1

        # Evaluate at intermediate point (x_mid, sigma_mid)
        timestep_2 = sigma_mid * 1000  # Convert sigma to timestep
        k2 = model(x_mid, timestep_2, **extra_args)

        # Apply BONGMATH iteration if enabled
        if self.use_bongmath:
            sample, k1, k2 = self.bongmath_iteration(
                sample, sigma, sigma_next, h, k1, k2,
                model, extra_args, iterations=100
            )

        # Final RK update using both stages
        prev_sample = sample + (sigma_next - sigma) * (b1 * k1 + b2 * k2)

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

        timestep_id = torch.argmin((self.timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma = self.sigmas[timestep_id].reshape(-1, 1, 1, 1)

        # Flow matching forward process: x_t = (1 - sigma) * x_0 + sigma * noise
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample.type_as(noise)