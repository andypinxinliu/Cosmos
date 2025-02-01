
![Cosmos Logo](assets/cosmos-logo.png)

--------------------------------------------------------------------------------
### [Website](https://www.nvidia.com/en-us/ai/cosmos/) | [HuggingFace](https://huggingface.co/collections/nvidia/cosmos-6751e884dc10e013a0a0d8e6) | [GPU-free Preview](https://build.nvidia.com/explore/discover) | [Paper](https://arxiv.org/abs/2501.03575) | [Paper Website](https://research.nvidia.com/labs/dir/cosmos1/)

[NVIDIA Cosmos](https://www.nvidia.com/cosmos/) is a developer-first world foundation model platform designed to help Physical AI developers build their Physical AI systems better and faster. Cosmos contains


## Key Features

- [Pre-trained Diffusion-based world foundation models](cosmos1/models/diffusion/README.md) for Text2World and Video2World generation where a user can generate visual simulation based on text prompts and video prompts.
- [Pre-trained Autoregressive-based world foundation models](cosmos1/models/autoregressive/README.md) for Video2World generation where a user can generate visual simulation based on video prompts and optional text prompts.
- [Video tokenizers](https://github.com/NVIDIA/Cosmos-Tokenizer) for tokenizing videos into continuous tokens (latent vectors) and discrete tokens (integers) efficiently and effectively.
- Video curation pipeline for building your own video dataset. [Coming soon]
- [Post-training scripts](cosmos1/models/POST_TRAINING.md) via NeMo Framework to post-train the pre-trained world foundation models for various Physical AI setup.
- Pre-training scripts via NeMo Framework for building your own world foundation model. [[Diffusion](https://github.com/NVIDIA/NeMo/tree/main/nemo/collections/diffusion)] [[Autoregressive](https://github.com/NVIDIA/NeMo/tree/main/nemo/collections/multimodal_autoregressive)] [[Tokenizer](https://github.com/NVIDIA/NeMo/tree/main/nemo/collections/diffusion/vae)].

## Environment Setup

There could be problems with transformer engine, not sure if it now works
```bash
conda create -n cosmos python=3.10 # must be greater or equal to 3.10
conda install -c conda-forge cxx-compiler
conda install gcc=9 # must be greater or equal to 9
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install nvidia/label/cuda-12.1.1::cuda-toolkit
conda install nvidia/label/cuda-12.1.1::cuda-nvcc
conda install anaconda::cudnn
pip install transformer_engine[pytorch]
pip install -r requirements.txt

```


## Download Checkpoints

1. Generate a [Hugging Face](https://huggingface.co/settings/tokens) access token. Set the access token to 'Read' permission (default is 'Fine-grained').

2. Log in to Hugging Face with the access token:

```bash
huggingface-cli login
```

3. Request access to Mistral AI's Pixtral-12B model by clicking on `Agree and access repository` on [Pixtral's Hugging Face model page](https://huggingface.co/mistralai/Pixtral-12B-2409). This step is required to use Pixtral 12B for the Video2World prompt upsampling task.

4. Download the Cosmos model weights from [Hugging Face](https://huggingface.co/collections/nvidia/cosmos-6751e884dc10e013a0a0d8e6):

```bash
python cosmos1/scripts/download_diffusion.py --model_sizes 7B 14B --model_types Text2World Video2World
```

5. The downloaded files should be in the following structure:

```
checkpoints/
├── Cosmos-1.0-Diffusion-7B-Text2World
│   ├── model.pt
│   └── config.json
├── Cosmos-1.0-Diffusion-14B-Text2World
│   ├── model.pt
│   └── config.json
├── Cosmos-1.0-Diffusion-7B-Video2World
│   ├── model.pt
│   └── config.json
├── Cosmos-1.0-Diffusion-14B-Video2World
│   ├── model.pt
│   └── config.json
├── Cosmos-1.0-Tokenizer-CV8x8x8
│   ├── decoder.jit
│   ├── encoder.jit
│   └── mean_std.pt
├── Cosmos-1.0-Prompt-Upsampler-12B-Text2World
│   ├── model.pt
│   └── config.json
├── Pixtral-12B
│   ├── model.pt
│   ├── config.json
└── Cosmos-1.0-Guardrail
    ├── aegis/
    ├── blocklist/
    ├── face_blur_filter/
    └── video_content_safety_filter/
```


### Inference


```bash
PROMPT="A sleek, humanoid robot stands in a vast warehouse filled with neatly stacked cardboard boxes on industrial shelves. \
The robot's metallic body gleams under the bright, even lighting, highlighting its futuristic design and intricate joints. \
A glowing blue light emanates from its chest, adding a touch of advanced technology. The background is dominated by rows of boxes, \
suggesting a highly organized storage system. The floor is lined with wooden pallets, enhancing the industrial setting. \
The camera remains static, capturing the robot's poised stance amidst the orderly environment, with a shallow depth of \
field that keeps the focus on the robot while subtly blurring the background for a cinematic effect."

# Example using 7B model
PYTHONPATH=$(pwd) python cosmos1/models/diffusion/inference/text2world.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-1.0-Diffusion-7B-Text2World \
    --prompt "$PROMPT" \
    --offload_prompt_upsampler \
    --video_save_name Cosmos-1.0-Diffusion-7B-Text2World
```

<video src="https://github.com/user-attachments/assets/db7bebfe-5314-40a6-b045-4f6ce0a87f2a">
  Your browser does not support the video tag.
</video>

We also offer [multi-GPU inference](cosmos1/models/diffusion/nemo/inference/README.md) support for Diffusion Text2World WFM models through NeMo Framework.

### Post-training

NeMo Framework provides GPU accelerated post-training with general post-training for both [diffusion](cosmos1/models/diffusion/nemo/post_training/README.md) and [autoregressive](cosmos1/models/autoregressive/nemo/post_training/README.md) models, with other types of post-training coming soon.

## License and Contact

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

NVIDIA Cosmos source code is released under the [Apache 2 License](https://www.apache.org/licenses/LICENSE-2.0).

NVIDIA Cosmos models are released under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license). For a custom license, please contact [cosmos-license@nvidia.com](mailto:cosmos-license@nvidia.com).
