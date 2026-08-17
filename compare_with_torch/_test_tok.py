import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokenizer import load_charbpe_from_file

t = load_charbpe_from_file("bytebpe.json")
assert t is not None, "char bpe tokenizer failed to load"
print("vocab_size =", t.vocab_size())
print("bos =", t.bos_id(), "eos =", t.eos_id(), "pad =", t.PAD_ID)

ids = t.encode("Once upon a time there was a little girl named Lily who loved to read stories.")
print("n_tokens =", len(ids))
print("min/max id =", min(ids), max(ids))
print("sample ids =", ids[:20])
print("decode =", t.decode(ids)[:80])
