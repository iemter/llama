# LLaMA 

This repository is intended as a minimal, hackable and readable example to load [LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/) ([arXiv](https://arxiv.org/abs/2302.13971v1)) models and run inference.
In order to download the checkpoints and tokenizer, fill this [google form](https://forms.gle/jk851eBVbX1m5TAv5)

## Setup

In a conda env with pytorch / cuda available, run:
```
pip install -r requirements.txt
```
Then in this repository:
```
pip install -e .
```

## Installation on Windows in detail
1. Download [CUDA](https://developer.nvidia.com/cuda-downloads) (12.1 works) and install 
2. Download [CuDNN](https://developer.nvidia.com/rdp/cudnn-download) and follow the [installation guidelines](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)
3. [Install PyTorch](https://pytorch.org/get-started/locally/), the version for CUDA 11.8 works for CUDA 12.1 as well
4. Make a new Anaconda environment with Python 3.11
5. Install sentencepiece using 
```
pip install sentencepiece-0.1.98-cp311-cp311-win_amd64.whl
```
(See [here](https://github.com/google/sentencepiece/issues/810))

6. Run
```
pip install -r requirements.txt 
```

Now everything should work 😸

## Download

Once your request is approved, you will receive links to download the tokenizer and model files.
Edit the `download.sh` script with the signed url provided in the email to download the model weights and tokenizer.

Alternatively, there are other ways to download the model you can find [here](https://github.com/facebookresearch/llama/issues/149)

## Inference

The provided `example.py` can be run on a single or multi-gpu node with `torchrun` and will output completions for two pre-defined prompts. Using `TARGET_FOLDER` as defined in `download.sh`:
```
torchrun --nproc_per_node MP example.py --ckpt_dir $TARGET_FOLDER/model_size --tokenizer_path $TARGET_FOLDER/tokenizer.model
```

Different models require different MP values:

|  Model | MP |
|--------|----|
| 7B     | 1  |
| 13B    | 2  |
| 33B    | 4  |
| 65B    | 8  |

You can also use `inference.py` with a lot of args on a single GPU. 
The 7B model works on a Single NVIDIA 4070Ti GPU. 

An inference-script is provided with inference.py. You need to add the folder with the model checkpoints using the --model_path argument and the tokenizer using the --tokenizer_path. 

You can also change parameters such as `temperature`, `top_p`, `max_seq_len`, `max_batch_size` or `max_gen_len` if you so desire.
## FAQ

- [1. The download.sh script doesn't work on default bash in MacOS X](FAQ.md#1)
- [2. Generations are bad!](FAQ.md#2)
- [3. CUDA Out of memory errors](FAQ.md#3)
- [4. Other languages](FAQ.md#4)

## Reference

LLaMA: Open and Efficient Foundation Language Models -- https://arxiv.org/abs/2302.13971

```
@article{touvron2023llama,
  title={LLaMA: Open and Efficient Foundation Language Models},
  author={Touvron, Hugo and Lavril, Thibaut and Izacard, Gautier and Martinet, Xavier and Lachaux, Marie-Anne and Lacroix, Timoth{\'e}e and Rozi{\`e}re, Baptiste and Goyal, Naman and Hambro, Eric and Azhar, Faisal and Rodriguez, Aurelien and Joulin, Armand and Grave, Edouard and Lample, Guillaume},
  journal={arXiv preprint arXiv:2302.13971},
  year={2023}
}
```

## Model Card
See [MODEL_CARD.md](MODEL_CARD.md)

## License
See the [LICENSE](LICENSE) file.
