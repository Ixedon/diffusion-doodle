from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler, AutoencoderKL
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
from controlnet_aux.pidi import PidiNetDetector
from functools import lru_cache
from PIL import Image

import gradio as gr
import numpy as np
import torch
import os



# set up the device and half percision
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16

###########################################################################
######          Image demo section  (first tab)            ################
###########################################################################

@lru_cache(maxsize=1)
def load_stable_diffusion_pipeline() -> tuple[StableDiffusionXLAdapterPipeline, PidiNetDetector]:
    # load the adapter
    adapter = T2IAdapter.from_pretrained(
    "TencentARC/t2i-adapter-sketch-sdxl-1.0", torch_dtype=torch_dtype, variant="fp16"
    )

    # load the SD models
    model_id = 'stabilityai/stable-diffusion-xl-base-1.0'
    euler_a = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    vae=AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch_dtype)
    
    # Create the pipeline
    pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
        model_id, vae=vae, adapter=adapter, scheduler=euler_a, torch_dtype=torch_dtype, variant="fp16", 
    )
    pipe.enable_model_cpu_offload()  # handles the moving of the model to GPU
    pipe.enable_xformers_memory_efficient_attention()

    # load the preprocessing network
    pidinet = PidiNetDetector.from_pretrained("lllyasviel/Annotators").to(device)
    
    return pipe, pidinet

@lru_cache(maxsize=1)
def load_voice_transcription_pipeline(whisper_model_name = "openai/whisper-small"):
    # Load the audio transcibtion models
    whisper_processor = AutoProcessor.from_pretrained(whisper_model_name)
    whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(whisper_model_name,
                                                            torch_dtype=torch_dtype,
                                                            use_safetensors=True).to(device)

    pipe = pipeline("automatic-speech-recognition",
                            model=whisper_model,
                            tokenizer=whisper_processor.tokenizer,
                            feature_extractor= whisper_processor.feature_extractor,
                            max_new_tokens = 128,
                            chunk_length_s=30,
                            torch_dtype=torch_dtype,
                            device=device)
    return pipe

def run_diffusion_pipeline(prompt, cond_img, seed) -> Image.Image:
    # Preprocess the image and create the conditioned image
    sd_pipe, pidinet = load_stable_diffusion_pipeline()
    pid_image = pidinet(cond_img, detect_resolution=1024, image_resolution=1024, apply_filter=True)
    
    output = sd_pipe(
                        prompt,
                        image = pid_image,
                        negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
                        num_inference_steps=20,
                        adapter_conditioning_scale =0.7,
                        generator = torch.manual_seed(seed)
                    )
    
    return output[0][0]


def save_image_to_disk(user_name: str, prompt: str, seed: int, input_image: Image.Image, output_image: Image.Image):
    if user_name == "": user_name = "anon"
    directory = os.path.join("demo_outputs", user_name)

    if not os.path.exists(directory):
        os.makedirs(os.path.join(directory, "prompt"))
        os.makedirs(os.path.join(directory, "gen"))

    unique_filename = prompt.replace(" ", "_")
    unique_filename = "".join(c for c in unique_filename if c.isalnum() or c in ('.','_', '-')).rstrip() + "-" + str(seed)

    try:
        input_image.save(os.path.join(directory, "prompt", unique_filename+"-prompt.png")) 
        output_image.save(os.path.join(directory, "gen", unique_filename+"-gen.png")) 

    except (ValueError, OSError) as e:  # Catch the two errors PIL.Image.Image.save can throw
        print(f"Image not saved: {e}")

def run(user_name: str, prompt: str, input_image: Image.Image, sample: str|None) -> tuple[str, Image.Image] :
    img = img["composite"]

    # Get the prompt from the text trascription (if selected)
    if sample is not None and prompt == "":
        whisper_pipe = load_voice_transcription_pipeline()
        prompt = whisper_pipe(sample, generate_kwargs = {"language":"polish", "task": "translate"})["text"]   #TODO: set the language in program parameters

    seed = torch.random.seed()  # get random seed, so that the model will not generate detrminitically
    output_image = run_diffusion_pipeline(prompt, img, seed)
    
    save_image_to_disk(user_name, prompt, seed, input_image, output_image)

    return prompt, output_image


image_demo = gr.Interface(
    fn = run,
    inputs= ["text", "text", gr.ImageEditor(type="pil"), gr.Audio(type="filepath")],
    outputs = [gr.Text(label="prompt"), "image"])


###########################################################################
######          Audio demo section  (second tab)           ################
###########################################################################

audio_pipe = pipeline("text-to-audio", model="facebook/musicgen-small", device=device)

def music_pipe(text: str) -> tuple[int, np.array]:
    music = audio_pipe(text)
    audio_data = music["audio"] * 32768
    audio_data = audio_data.astype(np.int16, order='C').reshape((-1,1))

    music = (music["sampling_rate"], audio_data)
    
    return music


audio_demo = gr.Interface(fn=music_pipe, inputs="text", outputs="audio")


###########################################################################
######                 Running the demo                    ################
###########################################################################

demo = gr.TabbedInterface([image_demo, audio_demo], ["Doodle", "Audio"])

login_data = None
# login_data = ("user", "password")  # replace with your own or leave None for no login page


demo.launch(share=True, auth=login_data)