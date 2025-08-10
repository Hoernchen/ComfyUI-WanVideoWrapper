"""
Flow Matching DPM++ 3M SDE Scheduler
Based on ComfyUI's dpmpp_3m_sde implementation from k-diffusion
"""
import torch
import math
from typing import Optional


class FlowDPMPP3MSDEScheduler:
    """
    DPM++ 3M SDE scheduler for flow matching.
    Third-order multistep SDE solver.
    """
    
    def __init__(self, num_train_timesteps=1000, shift=3.0):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self._sigmas = None
        self._timesteps = None
        self.num_inference_steps = None
        self.eta = 1.0  # SDE noise strength
        self.s_noise = 1.0  # Noise scaling
        self.use_gpu = False  # GPU noise sampling flag
        
        # History for multistep
        self.denoised_1 = None
        self.denoised_2 = None
        self.h_1 = None
        self.h_2 = None
        
    def set_timesteps(self, num_inference_steps, device=None, shift=None, 
                      use_beta_sigmas=False, sigmas=None, **kwargs):
        """Set the timesteps for the scheduler."""
        if shift is not None:
            self.shift = shift
            
        if sigmas is not None:
            # Use custom sigmas if provided
            self.sigmas = torch.tensor(sigmas, device=device)
            self.timesteps = (self.sigmas * 1000).to(torch.int64).to(device)
            self.num_inference_steps = len(self.sigmas)
        else:
            # Create sigmas using shift transformation
            sigma_max = 1.0
            sigma_min = 0.003 / 1.002
            
            # Linear spacing in sigma space
            sigmas = torch.linspace(sigma_max, sigma_min, num_inference_steps + 1)
            
            # Apply shift transformation
            sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
            
            # Include zero at the end
            sigmas[-1] = 0.0
            
            self.sigmas = sigmas.to(device)
            self.timesteps = (sigmas[:-1] * self.num_train_timesteps).to(torch.int64).to(device)
            self.num_inference_steps = num_inference_steps
            
        # Reset history
        self.denoised_1 = None
        self.denoised_2 = None
        self.h_1 = None
        self.h_2 = None
    
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
    
    def _sigma_to_lambda(self, sigma):
        """Convert sigma to lambda (half-log SNR) for flow matching."""
        # In flow matching: lambda = -log(sigma / (1 - sigma))
        # This is equivalent to half-log SNR
        return -torch.log(sigma / (1 - sigma + 1e-10))
    
    def step(self, model_output, timestep, sample, generator=None, **kwargs):
        """
        Perform a single step of the DPM++ 3M SDE sampling.
        
        Args:
            model_output: The model's prediction (velocity/flow)
            timestep: Current timestep
            sample: Current sample
            generator: Random number generator for SDE noise
            
        Returns:
            prev_sample: The sample at the previous timestep
        """
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
            
        self.sigmas = self.sigmas.to(model_output.device)
        self.timesteps = self.timesteps.to(model_output.device)
        
        # Find the index of the current timestep
        if timestep.ndim == 0:
            timestep_id = torch.argmin((self.timesteps - timestep).abs(), dim=0)
        else:
            timestep_id = torch.argmin((self.timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        
        # Get current and next sigma
        sigma = self.sigmas[timestep_id]
        sigma_next = self.sigmas[timestep_id + 1]
        
        if sigma.ndim > 0:
            sigma = sigma.reshape(-1, 1, 1, 1)
            sigma_next = sigma_next.reshape(-1, 1, 1, 1)
        else:
            sigma = sigma.view(1, 1, 1, 1)
            sigma_next = sigma_next.view(1, 1, 1, 1)
        
        # Convert flow prediction to denoised prediction
        # In flow matching: x_0 = x_t - sigma_t * v_t
        denoised = sample - sigma * model_output
        
        if sigma_next == 0:
            # Final step
            return denoised
        
        # Compute lambda values (half-log SNR)
        lambda_s = self._sigma_to_lambda(sigma)
        lambda_t = self._sigma_to_lambda(sigma_next)
        h = lambda_t - lambda_s
        h_eta = h * (self.eta + 1)
        
        alpha_t = sigma_next * torch.exp(lambda_t)
        
        # Base update
        x = sigma_next / sigma * torch.exp(-h * self.eta) * sample + \
            alpha_t * (1 - torch.exp(-h_eta)) * denoised
        
        if self.h_2 is not None and self.denoised_1 is not None and self.denoised_2 is not None:
            # Third-order update (DPM++ 3M SDE)
            r0 = self.h_1 / h
            r1 = self.h_2 / h
            d1_0 = (denoised - self.denoised_1) / r0
            d1_1 = (self.denoised_1 - self.denoised_2) / r1
            d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
            d2 = (d1_0 - d1_1) / (r0 + r1)
            
            phi_2 = (torch.exp(-h_eta) - 1) / h_eta + 1
            phi_3 = phi_2 / h_eta - 0.5
            
            x = x + (alpha_t * phi_2) * d1 - (alpha_t * phi_3) * d2
            
        elif self.h_1 is not None and self.denoised_1 is not None:
            # Second-order update (DPM++ 2M SDE)
            r = self.h_1 / h
            d = (denoised - self.denoised_1) / r
            phi_2 = (torch.exp(-h_eta) - 1) / h_eta + 1
            x = x + (alpha_t * phi_2) * d
        
        # Add SDE noise
        if self.eta > 0 and self.s_noise > 0:
            if generator is not None:
                # Generate noise on CPU (following WanVideoWrapper convention)
                # then move to GPU if needed
                noise_device = generator.device if hasattr(generator, 'device') else torch.device("cpu")
                noise = torch.randn(sample.shape, generator=generator, device=noise_device, dtype=torch.float32)
                # Move to sample device and dtype
                noise = noise.to(sample.device).to(sample.dtype)
            else:
                # No generator provided, use default random on sample device
                noise = torch.randn_like(sample)
            
            noise_scale = sigma_next * torch.sqrt(1 - torch.exp(-2 * h * self.eta)) * self.s_noise
            x = x + noise_scale * noise
        
        # Update history
        self.denoised_2 = self.denoised_1
        self.denoised_1 = denoised
        self.h_2 = self.h_1
        self.h_1 = h
        
        return x
    
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