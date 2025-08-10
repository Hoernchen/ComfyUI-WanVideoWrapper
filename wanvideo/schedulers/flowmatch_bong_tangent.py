"""
Flow Matching Bong Tangent Scheduler
Based on RES4LYF's bong_tangent scheduler - a two-stage tangent-based sigma schedule
"""
import torch
import math


class FlowBongTangentScheduler:
    """
    Bong Tangent scheduler for flow matching.
    Creates a two-stage tangent-based schedule with customizable pivots and slopes.
    """
    
    def __init__(self, num_train_timesteps=1000, shift=3.0):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self._sigmas = None
        self._timesteps = None
        self.num_inference_steps = None
        
    def get_bong_tangent_sigmas(self, steps, slope, pivot, start, end):
        """
        Generate tangent-based sigma values for a single stage.
        
        Args:
            steps: Number of steps for this stage
            slope: Slope of the tangent curve
            pivot: Pivot point for the tangent
            start: Starting sigma value
            end: Ending sigma value
            
        Returns:
            List of sigma values
        """
        # Calculate the tangent curve
        smax = ((2/math.pi) * math.atan(-slope * (0 - pivot)) + 1) / 2
        smin = ((2/math.pi) * math.atan(-slope * ((steps - 1) - pivot)) + 1) / 2
        
        srange = smax - smin
        sscale = start - end
        
        sigmas = []
        for x in range(steps):
            s = ((2/math.pi) * math.atan(-slope * (x - pivot)) + 1) / 2
            normalized = (s - smin) / srange
            sigma = normalized * sscale + end
            sigmas.append(sigma)
            
        return sigmas
    
    def set_timesteps(self, num_inference_steps, device=None, shift=None,
                      start=1.0, middle=0.5, end=0.0, 
                      pivot_1=0.6, pivot_2=0.6, 
                      slope_1=0.2, slope_2=0.2,
                      use_beta_sigmas=False, sigmas=None, **kwargs):
        """
        Set the timesteps for the scheduler using bong tangent method.
        
        Args:
            num_inference_steps: Number of inference steps
            device: Device to place tensors on
            shift: Shift parameter (overrides instance shift if provided)
            start: Starting sigma value (default 1.0)
            middle: Middle sigma value where stages meet (default 0.5)
            end: Ending sigma value (default 0.0)
            pivot_1: Pivot point for first stage (0-1, default 0.6)
            pivot_2: Pivot point for second stage (0-1, default 0.6)
            slope_1: Slope for first stage (default 0.2)
            slope_2: Slope for second stage (default 0.2)
            use_beta_sigmas: Ignored (for compatibility)
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
            # Generate bong tangent schedule
            steps = num_inference_steps + 2  # Add 2 as in original
            
            # Calculate pivot points and midpoint
            midpoint = int((steps * pivot_1 + steps * pivot_2) / 2)
            pivot_1_steps = int(steps * pivot_1)
            pivot_2_steps = int(steps * pivot_2)
            
            # Adjust slopes based on number of steps
            slope_1_adj = slope_1 / (steps / 40)
            slope_2_adj = slope_2 / (steps / 40)
            
            # Calculate stage lengths
            stage_2_len = steps - midpoint
            stage_1_len = steps - stage_2_len
            
            # Generate sigmas for each stage
            tan_sigmas_1 = self.get_bong_tangent_sigmas(stage_1_len, slope_1_adj, 
                                                         pivot_1_steps, start, middle)
            tan_sigmas_2 = self.get_bong_tangent_sigmas(stage_2_len, slope_2_adj, 
                                                         pivot_2_steps - stage_1_len, 
                                                         middle, end)
            
            # Combine stages (remove last element of first stage to avoid duplicate)
            tan_sigmas_1 = tan_sigmas_1[:-1]
            tan_sigmas = tan_sigmas_1 + tan_sigmas_2
            
            # Apply shift transformation for flow matching
            sigmas = torch.tensor(tan_sigmas, device=device)
            sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
            
            # Don't append zero for flow matching (following SGM uniform pattern)
            self.sigmas = sigmas[:-1]  # Remove the last element
            self.timesteps = (self.sigmas * self.num_train_timesteps).to(torch.int64)
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
    
    def step(self, model_output, timestep, sample, generator=None, return_dict=True, **kwargs):
        """
        Predict the sample at the previous timestep.
        This is a simple Euler step for compatibility.
        
        Args:
            model_output: Direct output from the model (velocity/flow prediction)
            timestep: Current discrete timestep
            sample: Current sample
            generator: Random number generator (not used)
            return_dict: Whether to return a dict
            
        Returns:
            prev_sample: Sample at the previous timestep
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
        
        # Simple Euler step for flow matching
        # dx/dt = v(x, t), so x_next = x + dt * v
        dt = sigma_next - sigma
        prev_sample = sample + dt * model_output
        
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