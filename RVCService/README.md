<div align="center">

<h1>RVCService</h1>

## Prerequisites
+ Python 3.8 (recommened) or higher 
+ Anaconda or miniconda that suitable with your python version.
+ Pip 23.2.1 or higher

## Preparing the environment


(Windows)
First we need create an anaconda environment that work with python 3.8 and activate it:
```bash
conda create -n rvc python=3.8
conda activate rvc
```
Second install PyTorch-related core dependencies. 
For CPU you can run:
```bash
# Install PyTorch-related core dependencies for cpu
pip install torch torchvision torchaudio
```
For Nvidia and CUDA machine, you can run:
```bash
# Or install PyTorch-related core dependencies for cpu
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
Note: For better performance we recommend to use machine with Nvidia graphic card with CUDA 11.8. If you want to check your CUDA version, you can use this command:

```bash
nvcc --version
```

If you do not have CUDA, you can follow this link to install it https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/. For install CUDA 11.8, you can download it in here: https://developer.nvidia.com/cuda-11-8-0-download-archive

Finally, install other dependencies.
```bash
pip install -r requirements.txt
```

## Usage
Open a Command Prompt, ensure your conda environment your created above is activated. If it deactivated, activate it with command:

```bash
conda activate rvc
```
Then excute the audio-processing.bat file

```bash
.\audio-processing.bat
```

If you see these lines in your Command Prompt. The service is ready to use now.

```bash
2024-04-03 22:39:23 | INFO | faiss.loader | Loading faiss with AVX2 support.
2024-04-03 22:39:23 | INFO | faiss.loader | Successfully loaded faiss with AVX2 support.
2024-04-03 22:39:27 | INFO | configs.config | Found GPU NVIDIA GeForce RTX 3050 Laptop GPU
2024-04-03 22:39:27 | INFO | configs.config | Half-precision floating-point: True, device: cuda:0
Loading...
Index search enabled
2024-04-03 22:39:33 | INFO | faiss.loader | Loading faiss with AVX2 support.
2024-04-03 22:39:33 | INFO | faiss.loader | Successfully loaded faiss with AVX2 support.
Load check point
2024-04-03 22:39:35 | INFO | fairseq.tasks.hubert_pretraining | current directory is G:\.NET\TmaRVC_v2\TheST\RVCService
2024-04-03 22:39:35 | INFO | fairseq.tasks.hubert_pretraining | HubertPretrainingTask Config {'_name': 'hubert_pretraining', 'data': 'metadata', 'fine_tuning': False, 'labels': ['km'], 'label_dir': 'label', 'label_rate': 50.0, 'sample_rate': 16000, 'normalize': False, 'enable_padding': False, 'max_keep_size': None, 'max_sample_size': 250000, 'min_sample_size': 32000, 'single_target': False, 'random_crop': True, 'pad_audio': False}
2024-04-03 22:39:35 | INFO | fairseq.models.hubert.hubert | HubertModel Config: {'_name': 'hubert', 'label_rate': 50.0, 'extractor_mode': default, 'encoder_layers': 12, 'encoder_embed_dim': 768, 'encoder_ffn_embed_dim': 3072, 'encoder_attention_heads': 12, 'activation_fn': gelu, 'layer_type': transformer, 'dropout': 0.1, 'attention_dropout': 0.1, 'activation_dropout': 0.0, 'encoder_layerdrop': 0.05, 'dropout_input': 0.1, 'dropout_features': 0.1, 'final_dim': 256, 'untie_final_proj': True, 'layer_norm_first': False, 'conv_feature_layers': '[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2', 'conv_bias': False, 'logit_temp': 0.1, 'target_glu': False, 'feature_grad_mult': 0.1, 'mask_length': 10, 'mask_prob': 0.8, 'mask_selection': static, 'mask_other': 0.0, 'no_mask_overlap': False, 'mask_min_space': 1, 'mask_channel_length': 10, 'mask_channel_prob': 0.0, 'mask_channel_selection': static, 'mask_channel_other': 0.0, 'no_mask_channel_overlap': False, 'mask_channel_min_space': 1, 'conv_pos': 128, 'conv_pos_groups': 16, 'latent_temp': [2.0, 0.5, 0.999995], 'skip_masked': False, 'skip_nomask': False, 'checkpoint_activations': False, 'required_seq_len_multiple': 2, 'depthwise_conv_kernel_size': 31, 'attn_type': '', 'pos_enc_type': 'abs', 'fp16': False}
C:\Users\admin\miniconda3\envs\theST\lib\site-packages\torch\nn\utils\weight_norm.py:28: UserWarning: torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm.
  warnings.warn("torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm.")
Hosted on 127.0.0.1:6666
======================================
```