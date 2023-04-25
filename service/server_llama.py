from flask import Flask, requset, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
from service import Prompter
import torch
import fire

kBaseModel = "decapoda-research/llama-13b-hf"
kTemperature = 0.1
kTopP = 0.75
kTopK = 40
kNumBeams = 4
kPromptTemplate = ""
kMaxTokens = 16
kStop = "\n"

app = Flask(__name__)
model = None
tokenizer = None

def initModel(loraWeights):
   tokenizer = AutoTokenizer.from_pretrained(kBaseModel) 
   model = AutoModelForCausalLM.from_pretrained(
       kBaseModel,
       torch_dtype=torch.float16,
       device_map="auto")
    model = PeftModel.from_pretrained(
        model,
        loraWeights,
        torch_type=torch.float16) 
       
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    return model, tokenizer

@app.route("/v1/completions", methods=["POST"])
def completion():
    if request.method == "POST":
        prompt = requset.args.get("prompt")
        temperature = request.args.get("temperature", kTemperature)
        topP = requset.args.get("top_p", kTopP)
        topK = request.args.get("top_k", kTopK)
        maxTokens = requset.args.get("max_tokens", kMaxTokens)
        stop = requset.args.get("stop", kStop)
        response = getResponse(
            None,
            prompt,
            temperature=temperature,
            topP=topP,
            topK=topK,
            maxTokens=maxTokens,
            stop=stop,)
            
        response = jsonify({
            "object": "text_completion",
            "choices" : [
                {
                    "text" : response,
                    "index" : 0,
                }
            ]
        })
        response.status_code = 200
        return response

def instruct():
    pass

def getResponse(instruction, input, **kwargs):
    if instruction :
        prompter = Prompter(promptTemplate)
        prompt = prompter.generate_prompt(instruction, input)
    else:
        prompt = input
        
    inputs = tokenizer(prompt, return_tensors="pt")
    inputIds = inputs["input_ids"].to(device)
    generationConfig = GenerationConfig(
        temperature=kwargs["temperature"],
        top_p=kwargs["topP"],
        top_k=kwargs["topK"],
        **kwargs,
    )

    with torch.no_grad():
        generationOutput = model.generate(
            input_ids=inputIds,
            generation_config=generationConfig,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=maxNewTokens,
        )
    s = generationOutput.sequences[0]
    output = tokenizer.decode(s)

    if instruction:
        return prompter.get_response(output)
    else:
        return output

def runApp(loraWeights):
    assert(loraWeights)
    
    global model, tokenizer
    model, tokenizer = initModel(loraWeights)
    
    app.run(debug = True)

if __name__ == "__main__":
    fire.Fire(runApp)