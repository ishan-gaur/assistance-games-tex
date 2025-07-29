#!/usr/bin/env python3
import torch
import math
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

def calculate_perplexity(model_name, context, message):
    """Calculate true perplexity using HuggingFace model logits."""

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize
    context_tokens = tokenizer.encode(context, return_tensors="pt")
    full_text = context + message
    full_tokens = tokenizer.encode(full_text, return_tensors="pt")

    # Get message tokens
    context_len = context_tokens.shape[1]
    message_tokens = full_tokens[:, context_len:]

    if message_tokens.shape[1] == 0:
        return {"error": "No message tokens found"}

    # Get logits and calculate probabilities
    with torch.no_grad():
        logits = model(full_tokens).logits

    # Get log probabilities for message tokens
    pred_logits = logits[0, context_len-1:-1]  # Logits that predict message tokens
    log_probs = torch.nn.functional.log_softmax(pred_logits, dim=-1)
    token_log_probs = log_probs.gather(1, message_tokens[0].unsqueeze(1)).squeeze(1)

    # Calculate perplexity
    avg_log_prob = token_log_probs.mean().item()
    perplexity = math.exp(-avg_log_prob)

    return {
        "model": model_name,
        "perplexity": perplexity,
        "avg_log_prob": avg_log_prob,
        "num_tokens": len(message_tokens[0])
    }

def main():
    parser = argparse.ArgumentParser(description="Calculate perplexity with HuggingFace models")
    parser.add_argument("--model", default="gpt2", help="HuggingFace model name")
    parser.add_argument("--context", required=True, help="Context text")
    parser.add_argument("--message", required=True, help="Message to evaluate")

    args = parser.parse_args()

    result = calculate_perplexity(args.model, args.context, args.message)

    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Perplexity: {result['perplexity']:.4f}")

if __name__ == "__main__":
    main()
