"""
Flow Matching SGM Uniform Scheduler
Based on ComfyUI's sgm_uniform scheduler implementation
"""
import torch


class FlowSGMUniformScheduler:
    """
    SGM Uniform scheduler for flow matching.
    Creates linear timesteps without appending zero at the end.
    """
    
    def __init__(self, num_train_timesteps=1000, shift=3.0):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self._sigmas = None
        self._timesteps = None
        self.num_inference_steps = None
    
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
            # Create linear sigmas (SGM uniform behavior)
            # Start from 1 and go down to sigma_min, but don't include 0
            sigma_max = 1.0
            sigma_min = 0.003 / 1.002  # Small but non-zero
            
            # Linear spacing without the final zero (SGM behavior)
            sigmas = torch.linspace(sigma_max, sigma_min, num_inference_steps + 1)[:-1]
            
            # Apply shift transformation
            sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
            
            self.sigmas = sigmas.to(device)
            self.timesteps = (sigmas * self.num_train_timesteps).to(torch.int64).to(device)
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
    
    def step(self, model_output, timestep, sample, **kwargs):
        """
        Perform a single step of the flow matching process.
        
        Args:
            model_output: The model's prediction (velocity/flow)
            timestep: Current timestep
            sample: Current sample
            
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
        if sigma.ndim > 0:
            sigma = sigma.reshape(-1, 1, 1, 1)
        else:
            sigma = sigma.view(1, 1, 1, 1)
            
        # Get next sigma (or 0 if at the end)
        if (timestep_id + 1 >= len(self.sigmas)).any() if timestep_id.ndim > 0 else timestep_id + 1 >= len(self.sigmas):
            sigma_next = torch.zeros_like(sigma)
        else:
            sigma_next = self.sigmas[timestep_id + 1]
            if sigma_next.ndim > 0:
                sigma_next = sigma_next.reshape(-1, 1, 1, 1)
            else:
                sigma_next = sigma_next.view(1, 1, 1, 1)
        
        # Flow matching step: x_{t-1} = x_t + v * (sigma_{t-1} - sigma_t)
        prev_sample = sample + model_output * (sigma_next - sigma)
        
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