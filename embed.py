"""ESM-2 mean-pooled embeddings, cached to disk keyed by accession."""
import os
import numpy as np
import torch

MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
CACHE_PATH = "data/embeddings.npz"
BATCH_SIZE = 8
MAX_LEN = 1024  # 1022 residues + CLS + EOS special tokens


def _embed_batch(model, tokenizer, seqs):
    enc = tokenizer(seqs, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN)
    with torch.no_grad():
        out = model(**enc)
    hidden = out.last_hidden_state  # (batch, tokens, 320)
    # exclude special tokens (cls/eos) as well as padding from the mean-pool
    special_mat = torch.tensor(
        [tokenizer.get_special_tokens_mask(ids.tolist(), already_has_special_tokens=True) for ids in enc["input_ids"]]
    )
    mask = (enc["attention_mask"] * (1 - special_mat)).unsqueeze(-1)
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return (summed / counts).numpy()


def compute_or_load_embeddings(df):
    """Keyed cache: rebuild the (N, 320) matrix in df order from an accession->vector
    dict, computing only sequences missing from the cache. This guarantees embeddings
    never misalign with labels even if the dataset order changes between runs."""
    cache = {}
    if os.path.exists(CACHE_PATH):
        npz = np.load(CACHE_PATH, allow_pickle=True)
        cache = dict(zip(npz["accessions"], npz["embeddings"]))

    missing = [(acc, seq) for acc, seq in zip(df.accession, df.sequence) if acc not in cache]
    if missing:
        print(f"Embedding {len(missing)} sequences with {MODEL_NAME} (cached: {len(cache)})...")
        from transformers import AutoTokenizer, AutoModel

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME)
        model.eval()
        for i in range(0, len(missing), BATCH_SIZE):
            batch = missing[i : i + BATCH_SIZE]
            accs, seqs = zip(*batch)
            vecs = _embed_batch(model, tokenizer, list(seqs))
            for acc, vec in zip(accs, vecs):
                cache[acc] = vec.astype(np.float32)
            if i % (BATCH_SIZE * 20) == 0:
                print(f"  {i + len(batch)}/{len(missing)}")
        accs_arr = np.array(list(cache.keys()))
        emb_arr = np.stack(list(cache.values()))
        np.savez(CACHE_PATH, accessions=accs_arr, embeddings=emb_arr)
    else:
        print(f"All {len(df)} embeddings loaded from cache at {CACHE_PATH}")

    return np.stack([cache[acc] for acc in df.accession])
