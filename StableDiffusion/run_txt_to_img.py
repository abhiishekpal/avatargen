import modules.scripts
from modules import sd_samplers
from modules.generation_parameters_copypaste import create_override_settings_dict
from modules.processing import StableDiffusionProcessing, Processed, StableDiffusionProcessingTxt2Img, \
    StableDiffusionProcessingImg2Img, process_images
from modules.shared import opts, cmd_opts
import modules.shared as shared
import modules.processing as processing
from modules.ui import plaintext_to_html
import sys
import config as C
import modules.codeformer_model as codeformer
import modules.face_restoration
import modules.gfpgan_model as gfpgan
import modules.img2img

import modules.lowvram
import modules.paths
import modules.scripts
import modules.sd_hijack
import modules.sd_models
import modules.sd_vae
import modules.txt2img
import modules.script_callbacks
import modules.textual_inversion.textual_inversion
import modules.progress

import modules.ui
from modules import modelloader
from modules.shared import cmd_opts
import modules.hypernetworks.hypernetwork

from modules import shared, devices, sd_samplers, upscaler, extensions, localization, ui_tempdir, ui_extra_networks
import modules.codeformer_model as codeformer
import modules.face_restoration
import modules.gfpgan_model as gfpgan
import modules.img2img

import modules.lowvram
import modules.paths
import modules.scripts
import modules.sd_hijack
import modules.sd_models
import modules.sd_vae
import modules.txt2img
import modules.script_callbacks
import modules.textual_inversion.textual_inversion
import modules.progress

import modules.ui
from modules import modelloader
from modules.shared import cmd_opts
import modules.hypernetworks.hypernetwork

from modules import import_hook, errors, extra_networks, ui_extra_networks_checkpoints
from modules import extra_networks_hypernet, ui_extra_networks_hypernets, ui_extra_networks_textual_inversion
from modules.call_queue import wrap_queued_call, queue_lock, wrap_gradio_gpu_call

extensions.list_extensions()
localization.list_localizations(cmd_opts.localizations_dir)

# if cmd_opts.ui_debug_mode:
#     shared.sd_upscalers = upscaler.UpscalerLanczos().scalers
#     modules.scripts.load_scripts()
#     return

modelloader.cleanup_models()
modules.sd_models.setup_model()
codeformer.setup_model(cmd_opts.codeformer_models_path)
gfpgan.setup_model(cmd_opts.gfpgan_models_path)
shared.face_restorers.append(modules.face_restoration.FaceRestoration())

modelloader.list_builtin_upscalers()
modules.scripts.load_scripts()
modelloader.load_upscalers()

modules.sd_vae.refresh_vae_list()

modules.textual_inversion.textual_inversion.list_textual_inversion_templates()

try:
    modules.sd_models.load_model()
except Exception as e:
    errors.display(e, "loading stable diffusion model")
    print("", file=sys.stderr)
    print("Stable diffusion model failed to load, exiting", file=sys.stderr)
    exit(1)

shared.opts.data["sd_model_checkpoint"] = shared.sd_model.sd_checkpoint_info.title

shared.opts.onchange("sd_model_checkpoint", wrap_queued_call(lambda: modules.sd_models.reload_model_weights()))
shared.opts.onchange("sd_vae", wrap_queued_call(lambda: modules.sd_vae.reload_vae_weights()), call=False)
shared.opts.onchange("sd_vae_as_default", wrap_queued_call(lambda: modules.sd_vae.reload_vae_weights()), call=False)
shared.opts.onchange("temp_dir", ui_tempdir.on_tmpdir_changed)

shared.reload_hypernetworks()

ui_extra_networks.intialize()
ui_extra_networks.register_page(ui_extra_networks_textual_inversion.ExtraNetworksPageTextualInversion())
ui_extra_networks.register_page(ui_extra_networks_hypernets.ExtraNetworksPageHypernetworks())
ui_extra_networks.register_page(ui_extra_networks_checkpoints.ExtraNetworksPageCheckpoints())

extra_networks.initialize()
extra_networks.register_extra_network(extra_networks_hypernet.ExtraNetworkHypernet())



prompt = "random person  <lora:Abhishek:1> riding a motorbike"
negative_prompt = ""

p = StableDiffusionProcessingTxt2Img(
    sd_model=shared.sd_model,
    outpath_samples=C.output_dir_samples,
    outpath_grids=C.outputdir_grids,
    prompt=prompt,
    styles=C.prompt_styles,
    negative_prompt=negative_prompt,
    seed=C.seed,
    subseed=C.subseed,
    subseed_strength=C.subseed_strength,
    seed_resize_from_h=C.seed_resize_from_h,
    seed_resize_from_w=C.seed_resize_from_w,
    seed_enable_extras=C.seed_enable_extras,
    sampler_name=C.sampler_name,
    batch_size=C.batch_size,
    n_iter=C.n_iter,
    steps=C.steps,
    cfg_scale=C.cfg_scale,
    width=C.width,
    height=C.height,
    restore_faces=C.restore_faces,
    tiling=C.tiling,
    enable_hr=C.enable_hr,
    denoising_strength=C.denoising_strength,
    hr_scale=C.hr_scale,
    hr_upscaler=C.hr_upscaler,
    hr_second_pass_steps=C.hr_second_pass_steps,
    hr_resize_x=C.hr_resize_x,
    hr_resize_y=C.hr_resize_y,
    override_settings=C.override_settings,
)
args = [0, False, False, 'LoRA', 'None', 1, 1, 'LoRA', 'None', 1, 1, 'LoRA', 'None', 1, 1, 'LoRA', 'None', 1, 1, 'LoRA', 'None', 1, 1, 'Refresh models', False, False, 'positive', 'comma', 0, False, False, '', 1, '', 0, '', 0, '', True, False, False, False, 0]
p.scripts = modules.scripts.scripts_txt2img
p.script_args = args[:]

if cmd_opts.enable_console_prompts:
    print(f"\ntxt2img: {prompt}", file=shared.progress_print_out)

processed = modules.scripts.scripts_txt2img.run(p, *args)

if processed is None:
    print('Processed is None')
    processed = process_images(p)

p.close()

from PIL import Image
 
im1 = processed.images[0].save(r"C:\Users\91973\Documents\AIGRAM\test.png")

shared.total_tqdm.clear()
print(processed.images)
