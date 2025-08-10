import torch
from .fm_solvers import (FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps)
from .fm_solvers_unipc import FlowUniPCMultistepScheduler
from .basic_flowmatch import FlowMatchScheduler
from .flowmatch_pusa import FlowMatchSchedulerPusa
from .flowmatch_res_multistep import FlowMatchSchedulerResMultistep
from .scheduling_flow_match_lcm import FlowMatchLCMScheduler
from .flowmatch_sgm_uniform import FlowSGMUniformScheduler
from .flowmatch_dpmpp_2s_ancestral import FlowDPMPP2SAncestralScheduler
from .flowmatch_dpmpp_3m_sde import FlowDPMPP3MSDEScheduler
from .flowmatch_res_2s import FlowRes2sScheduler
from .flowmatch_bong_tangent import FlowBongTangentScheduler
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler, DEISMultistepScheduler

from ...utils import log

scheduler_list = [
    "unipc", "unipc/beta",
    "dpm++", "dpm++/beta",  # Legacy names
    "dpmpp_2m", "dpmpp_2m/beta",  # DPM++ 2nd order multistep
    "dpmpp_3m", "dpmpp_3m/beta",  # DPM++ 3rd order multistep (deterministic only)
    "dpmpp_2m_sde", "dpmpp_2m_sde/beta",  # DPM++ 2nd order multistep SDE
    "dpmpp_3m_sde", "dpmpp_3m_sde/beta",  # DPM++ 3rd order multistep SDE
    "dpmpp_sde", "dpmpp_sde/beta",  # DPM++ 1st order SDE
    "dpmpp_2m_sde_gpu", "dpmpp_2m_sde_gpu/beta",  # GPU-optimized 2M SDE
    "dpmpp_3m_sde_gpu", "dpmpp_3m_sde_gpu/beta",  # GPU-optimized 3M SDE
    "dpmpp_sde_gpu", "dpmpp_sde_gpu/beta",  # GPU-optimized 1st order SDE
    "dpmpp_2s_ancestral", "dpmpp_2s_ancestral/beta",  # 2nd order single-step ancestral
    "dpm++_sde", "dpm++_sde/beta",  # Legacy names
    "euler", "euler/beta",
    #"euler/accvideo",
    "deis",
    "lcm", "lcm/beta",
    "res_multistep",
    "flowmatch_causvid",
    "flowmatch_distill",
    "flowmatch_pusa",
    "multitalk",
    "sgm_uniform",
    "res_2s", "res_2s/beta",  # RES4LYF 2nd order with BONGMATH
    "bong_tangent", "bong_tangent/beta"  # Bong tangent sigma schedule
]

def get_scheduler(scheduler, steps, shift, device, transformer_dim, flowedit_args, denoise_strength, sigmas=None):
    timesteps = None
    if 'unipc' in scheduler:
        sample_scheduler = FlowUniPCMultistepScheduler(shift=shift)
        if sigmas is None:
            sample_scheduler.set_timesteps(steps, device=device, shift=shift, use_beta_sigmas=('beta' in scheduler))
        else:
            sample_scheduler.sigmas = sigmas.to(device)
            sample_scheduler.timesteps = (sample_scheduler.sigmas[:-1] * 1000).to(torch.int64).to(device)
            sample_scheduler.num_inference_steps = len(sample_scheduler.timesteps)

    elif scheduler in ['euler/beta', 'euler']:
        sample_scheduler = FlowMatchEulerDiscreteScheduler(shift=shift, use_beta_sigmas=(scheduler == 'euler/beta'))
        if flowedit_args: #seems to work better
            timesteps, _ = retrieve_timesteps(sample_scheduler, device=device, sigmas=get_sampling_sigmas(steps, shift))
        else:
            sample_scheduler.set_timesteps(steps, device=device, sigmas=sigmas[:-1].tolist() if sigmas is not None else None)
    # elif scheduler in ['euler/accvideo']:
    #     if steps != 50:
    #         raise Exception("Steps must be set to 50 for accvideo scheduler, 10 actual steps are used")
    #     sample_scheduler = FlowMatchEulerDiscreteScheduler(shift=shift, use_beta_sigmas=(scheduler == 'euler/beta'))
    #     sample_scheduler.set_timesteps(steps, device=device, sigmas=sigmas.tolist() if sigmas is not None else None)
    #     start_latent_list = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    #     sample_scheduler.sigmas = sample_scheduler.sigmas[start_latent_list]
    #     steps = len(start_latent_list) - 1
    #     sample_scheduler.timesteps = timesteps = sample_scheduler.timesteps[start_latent_list[:steps]]
    elif 'dpm++' in scheduler or 'dpmpp' in scheduler:
        # Parse DPM++ variant details
        solver_order = 2  # Default
        algorithm_type = "dpmsolver++"
        use_gpu = False

        # Handle specific variants
        if 'dpmpp_3m' in scheduler:
            solver_order = 3
            # For SDE variants, we need special handling since diffusers doesn't support 3rd order SDE
            # but ComfyUI's k-diffusion does
        elif 'dpmpp_2m' in scheduler:
            solver_order = 2
        elif 'dpmpp_sde' in scheduler and 'dpmpp_2m_sde' not in scheduler and 'dpmpp_3m_sde' not in scheduler:
            solver_order = 1  # First-order SDE
        elif 'dpmpp_2s_ancestral' in scheduler:
            # This needs special handling - we'll implement it separately
            pass

        # Check for SDE variants
        if 'sde' in scheduler:
            algorithm_type = "sde-dpmsolver++"

        # Check for GPU variants
        if '_gpu' in scheduler:
            use_gpu = True

        # Create scheduler with appropriate configuration
        if 'dpmpp_3m_sde' in scheduler:
            # Use custom implementation for 3M SDE
            sample_scheduler = FlowDPMPP3MSDEScheduler(shift=shift)
            sample_scheduler.use_gpu = use_gpu
            if sigmas is None:
                sample_scheduler.set_timesteps(steps, device=device, use_beta_sigmas=('beta' in scheduler))
            else:
                sample_scheduler.set_timesteps(steps, device=device, sigmas=sigmas[:-1].tolist())
        elif 'dpmpp_2s_ancestral' in scheduler:
            # Handle dpmpp_2s_ancestral
            sample_scheduler = FlowDPMPP2SAncestralScheduler(shift=shift)
            if sigmas is None:
                sample_scheduler.set_timesteps(steps, device=device, use_beta_sigmas=('beta' in scheduler))
            else:
                sample_scheduler.set_timesteps(steps, device=device, sigmas=sigmas[:-1].tolist())
        else:
            # Use diffusers implementation for other variants
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                shift=shift,
                algorithm_type=algorithm_type,
                solver_order=solver_order
            )
            if sigmas is None:
                sample_scheduler.set_timesteps(steps, device=device, use_beta_sigmas=('beta' in scheduler))
            else:
                sample_scheduler.sigmas = sigmas.to(device)
                sample_scheduler.timesteps = (sample_scheduler.sigmas[:-1] * 1000).to(torch.int64).to(device)
                sample_scheduler.num_inference_steps = len(sample_scheduler.timesteps)

            # Store GPU flag for later use in sampling
            sample_scheduler.use_gpu = use_gpu
    elif scheduler == 'deis':
        sample_scheduler = DEISMultistepScheduler(use_flow_sigmas=True, prediction_type="flow_prediction", flow_shift=shift)
        sample_scheduler.set_timesteps(steps, device=device)
        sample_scheduler.sigmas[-1] = 1e-6
    elif 'lcm' in scheduler:
        sample_scheduler = FlowMatchLCMScheduler(shift=shift, use_beta_sigmas=(scheduler == 'lcm/beta'))
        sample_scheduler.set_timesteps(steps, device=device, sigmas=sigmas[:-1].tolist() if sigmas is not None else None)
    elif 'flowmatch_causvid' in scheduler:
        if sigmas is not None:
            raise NotImplementedError("This scheduler does not support custom sigmas")
        if transformer_dim == 5120:
            denoising_list = [999, 934, 862, 756, 603, 410, 250, 140, 74]
        else:
            if steps != 4:
                raise ValueError("CausVid 1.3B schedule is only for 4 steps")
            denoising_list = [1000, 750, 500, 250]
        sample_scheduler = FlowMatchScheduler(num_inference_steps=steps, shift=shift, sigma_min=0, extra_one_step=True)
        sample_scheduler.timesteps = torch.tensor(denoising_list)[:steps].to(device)
        sample_scheduler.sigmas = torch.cat([sample_scheduler.timesteps / 1000, torch.tensor([0.0], device=device)])
    elif 'flowmatch_distill' in scheduler:
        if sigmas is not None:
            raise NotImplementedError("This scheduler does not support custom sigmas")
        sample_scheduler = FlowMatchScheduler(
            shift=shift, sigma_min=0.0, extra_one_step=True
        )
        sample_scheduler.set_timesteps(1000, training=True)

        denoising_step_list = torch.tensor([999, 750, 500, 250] , dtype=torch.long)
        temp_timesteps = torch.cat((sample_scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
        denoising_step_list = temp_timesteps[1000 - denoising_step_list]
        #print("denoising_step_list: ", denoising_step_list)

        if steps != 4:
            raise ValueError("This scheduler is only for 4 steps")

        sample_scheduler.timesteps = denoising_step_list[:steps].clone().detach().to(device)
        sample_scheduler.sigmas = torch.cat([sample_scheduler.timesteps / 1000, torch.tensor([0.0], device=device)])
    elif 'flowmatch_pusa' in scheduler:
        sample_scheduler = FlowMatchSchedulerPusa(
            shift=shift, sigma_min=0.0, extra_one_step=True
        )
        sample_scheduler.set_timesteps(steps, denoising_strength=denoise_strength, shift=shift, sigmas=sigmas[:-1].tolist() if sigmas is not None else None)
    elif scheduler == 'res_multistep':
        sample_scheduler = FlowMatchSchedulerResMultistep(shift=shift)
        sample_scheduler.set_timesteps(steps, denoising_strength=denoise_strength, sigmas=sigmas[:-1].tolist() if sigmas is not None else None)
    elif scheduler == 'sgm_uniform':
        sample_scheduler = FlowSGMUniformScheduler(shift=shift)
        if sigmas is None:
            sample_scheduler.set_timesteps(steps, device=device, shift=shift)
        else:
            sample_scheduler.set_timesteps(steps, device=device, shift=shift, sigmas=sigmas[:-1].tolist())
    elif 'res_2s' in scheduler:
        # RES 2S with two-stage evaluation and optional BONGMATH
        use_bongmath = True  # Could be configurable later
        sample_scheduler = FlowRes2sScheduler(shift=shift, use_bongmath=use_bongmath)
        if sigmas is None:
            sample_scheduler.set_timesteps(steps, device=device, shift=shift, use_beta_sigmas=('beta' in scheduler))
        else:
            sample_scheduler.set_timesteps(steps, device=device, shift=shift, sigmas=sigmas[:-1].tolist())
    elif 'bong_tangent' in scheduler:
        # Bong tangent sigma schedule with customizable parameters
        sample_scheduler = FlowBongTangentScheduler(shift=shift)
        if sigmas is None:
            # Could expose these as configurable parameters later
            sample_scheduler.set_timesteps(steps, device=device, shift=shift,
                                         start=1.0, middle=0.5, end=0.0,
                                         pivot_1=0.6, pivot_2=0.6,
                                         slope_1=0.2, slope_2=0.2,
                                         use_beta_sigmas=('beta' in scheduler))
        else:
            sample_scheduler.set_timesteps(steps, device=device, shift=shift, sigmas=sigmas[:-1].tolist())
    if timesteps is None:
        timesteps = sample_scheduler.timesteps
    return sample_scheduler, timesteps