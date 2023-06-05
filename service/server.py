from flask import Flask, request, jsonify
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer, AutoConfig
from prompter import Prompter
import torch
import fire

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

def initModel(
        configNameOrPath,
        tokenizerNameOrPath,
        modelNameOrPath, 
        loraWeights):
    config = AutoConfig.from_pretrained(configNameOrPath)
    tokenizer = AutoTokenizer.from_pretrained(tokenizerNameOrPath) 
    if not loraWeights:
        model = AutoModelForCausalLM.from_pretrained(
            modelNameOrPath,
            config=config,
            torch_dtype=torch.float16,
            device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            modelNameOrPath,
            config=config,
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
    if torch.__version__ >= "2":
        model = torch.compile(model)
    return model, tokenizer

@app.route("/v1/completions", methods=["POST"])
def completion():
    if request.method == "POST":
        prompt = request.get_json().get("prompt")
        print("prompt[%s]" % prompt)
        temperature = request.get_json().get("temperature", kTemperature)
        topP = request.get_json().get("top_p", kTopP)
        topK = request.get_json().get("top_k", kTopK)
        maxTokens = request.get_json().get("max_tokens", kMaxTokens)
        stop = request.get_json().get("stop", kStop)
        response = getResponse(
            None,
            prompt,
            temperature=temperature,
            topP=topP,
            topK=topK,
            maxTokens=maxTokens,
            stop=stop,)
            
        response = jsonify({
            "errno" : 0,
            "msg" : "success",
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
        prompter = Prompter("")
        prompt = prompter.generate_prompt(instruction, input)
    else:
        prompt = input
        
    inputs = tokenizer(prompt, return_tensors="pt")
    inputIds = inputs["input_ids"].to("cuda")
    generationConfig = GenerationConfig(
        top_p=kwargs["topP"],
        top_k=kwargs["topK"],
        num_beams=kNumBeams,
        **kwargs,
    )

    with torch.no_grad():
        generationOutput = model.generate(
            input_ids=inputIds,
            generation_config=generationConfig,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=kwargs["maxTokens"],
        )
    s = generationOutput.sequences[0]
    output = tokenizer.decode(s)

    if instruction:
        return prompter.get_response(output)
    else:
        return output

def runApp(
        config_name_or_path,
        tokenizer_name_or_path,
        model_name_or_path, 
        lora_weights=None):
    assert(config_name_or_path)
    assert(tokenizer_name_or_path)
    assert(model_name_or_path)

    global model, tokenizer
    model, tokenizer = initModel(
        config_name_or_path,
        tokenizer_name_or_path,
        model_name_or_path, 
        lora_weights)
    
    app.run(debug = True)

if __name__ == "__main__":
    fire.Fire(runApp)