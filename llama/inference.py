from generation import LLaMA
from pathlib import Path
from model_single import ModelArgs, Transformer
from tokenizer import Tokenizer
import torch
import json


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]

    checkpoint = torch.load(ckpt_path, map_location="cpu")

    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)
    generator = LLaMA(model, tokenizer)
    return generator


def run(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 1024,
    max_batch_size: int = 1,
    max_gen_len: int = 512,
):
    local_rank = 0
    world_size = 1
    generator = load(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
    )

    print(
        """
        ~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
        Remember when prompting: LLaMA is good for auto completion, which means it generates
        an appropriate continuation of your prompt. As such, it can not directly answer questions. 
        Instead of asking "What is the meaning of life?", try prompting "The meaning of life is".
        Have fun!
        ~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
        """
    )

    while True: 
        user_input = input("please enter a prompt (type exit now to exit):\n")
        if user_input == "exit now":
            break

        prompts = [user_input]
        print("Prompt:", prompts)
        results = generator.generate(
            prompts, max_gen_len=max_gen_len, temperature=temperature, top_p=top_p
        )
        print(
            """
             ∩~~∩ 
            ξ *x* ξ    -- (Hello! I am LLaMA, a Language Model for Autocompletion.)
            ξ  ~  ξ 
            ξ     ξ 
            ξ     “~~~~~() 
            ξ           ξ 
            ξ ξ ξ~*~ξ   ξ 
             ξ_ξξ_ξ  ξ_ξξ_ξ

            """
        )
        print("LLaMA says: ")
        print("=========================================\n")
        for result in results:
            print(result.strip())
        print("\n=========================================\n")

    print("Goodbye!")

def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default="/llama_data/7B")
    parser.add_argument(
        "--tokenizer_path", type=str, default="/llama_data/tokenizer.model"
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--max_batch_size", type=int, default=1)
    parser.add_argument("--max_gen_len", type=int, default=512)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    run(
        ckpt_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer_path,
        temperature=args.temperature,
        top_p=args.top_p,
        max_seq_len=args.max_seq_len,
        max_batch_size=args.max_batch_size,
        max_gen_len=args.max_gen_len,
    )
