from transformers import AutoModelForCausalLM, AutoTokenizer
import json

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B", trust_remote_code=True)
vocab = tokenizer.get_vocab()

fout = open("vocab.txt", "w")
for key in vocab.keys():
        try:
                pat = key.decode("utf-8").strip()
                if len(pat) == 0:
                        continue

                j = {"piece":pat}
                fout.write("%s\n" % json.dumps(j, ensure_ascii=False))
        except Exception:
                pass
fout.close()
