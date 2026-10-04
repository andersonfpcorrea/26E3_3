from dlv.clip_search import describe_tokens

with stopwatch.measure("tokenização"):
    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenisation = {
        "texts": TOKEN_TEXTS,
        "clip": describe_tokens(processor.tokenizer, TOKEN_TEXTS),
        "bert": describe_tokens(bert_tokenizer, TOKEN_TEXTS),
    }
print("limite de tokens do CLIP:", processor.tokenizer.model_max_length)
for name in ("clip", "bert"):
    for text, described in zip(TOKEN_TEXTS, tokenisation[name]):
        print(f"{name}: {text[:40]!r}: {len(described['tokens'])} posições, {described['padding']} de preenchimento")
        print("  tokens:", described["tokens"])
        print("  máscara de atenção:", described["attention_mask"])
