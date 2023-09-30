import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

kWorldSize=4
kModel="THUDM/chatglm2-6b"
kLocalModelPath=kModel
kFilepath="/mnt/data1/dev/data/common_crawl/202210/clean_docs1.txt"
kLimit=5000000
kMaxLen = 4000

kArgs = {
    "cache_dir": None,
    "revision": "main",
    "use_auth_token": "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",
    "trust_remote_code": True
}

tokenizer = AutoTokenizer.from_pretrained(
        kModel,
        torch_dtype=torch.bfloat16,
        **kArgs)

config = AutoConfig.from_pretrained(
        kModel,
        torch_dtype=torch.bfloat16,
        **kArgs)

model = AutoModelForCausalLM.from_pretrained(
    kLocalModelPath,
    config=config,
    torch_dtype=torch.bfloat16,
    **kArgs)

inputText = [
        "The quick brown fox jumps over the lazy dog",
        "这件艺术品是中华艺术的瑰宝",
        "跟隐马尔可夫模型通过联合分布进行建模不同，条件随机场试图对多个变量在给定观测值后的条件概率进行建模"]

def readlinesFromFile(filepath, limit):
    with open(filepath, "r") as file:
        lines = []
        for i in range(limit):
            line = file.readline()
            if not line:
                break
            lines.append([line.strip(), 0.0])
    return lines

inputText = readlinesFromFile(kFilepath, kLimit)

def calculatePpl(rank, localModel, text):
    inputIds = tokenizer.encode(text, return_tensors="pt").to(rank)
    output = localModel(
            inputIds,
            labels=inputIds,
            use_cache=False)
    loss = output.loss
    return torch.exp(loss)

def runInference(rank, worldSize):
    print("run_inference[%d]" % rank)
    dist.init_process_group("nccl", rank=rank, world_size=worldSize)

    fout = open("/mnt/data1/dev/github/distiller/data/result_%d.txt" % rank, "w")

    model.to(rank)
    model.eval()
    idx=0
    with torch.inference_mode():
        while idx < len(inputText):
            if idx % kWorldSize == rank:
                t0 = time.time()
                if len(inputText[idx][0]) <= kMaxLen:
                    inputText[idx][1] = calculatePpl(rank, model, inputText[idx][0])
                    fout.write("%f\t%s\n" % (inputText[idx][1], inputText[idx][0]))

                t1 = time.time()
                print("idx[%d] len[%d] ppl[%f] time[%f]" % (
                    idx,
                    len(inputText[idx][0]),
                    inputText[idx][1],
                    t1-t0))

            idx+=1

    fout.close()

def main():
    t0 = time.time()
    mp.spawn(runInference, args=(kWorldSize,), nprocs=kWorldSize, join=True)
    t1 = time.time()
    print("overall_time[%f]" % (t1-t0))

if __name__ == "__main__":
    main()
