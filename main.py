# Simple command line to ensure the model work
import json, copy
from llama_cpp import Llama

VICUNA_13B_PATH = "/Users/randalldorin/Developer/ggml-models/ggml-vicuna-13b-4bit-rev1.bin"
VICUNA_7B_PATH = "/Users/randalldorin/Developer/ggml-models/ggml-vicuna-7b-1.1-q4_0.bin"

ALPACA_7B_PATH = "/Users/randalldorin/Developer/ggml-models/ggml-alpaca-7b-q4.bin"
ALPACA_30B_PATH = "/Users/randalldorin/Developer/ggml-models/ggml-alpaca-lora-30B-q4_1.bin"

LLAMA_7B_PATH = "/Users/randalldorin/Developer/ggml-models/ggml-llama-7b-q4_0.bin"
LLAMA_13B_PATH = "/Users/randalldorin/Developer/ggml-models/ggml-llama-13b-q4_0.bin"

KOALA_13B_PATH = "/Users/randalldorin/Developer/ggml-models/koala-13B-4bit-128g.GGML.bin"

MODEL_PATH = KOALA_13B_PATH

Prompt="Question: What a good recipe for apple pie? Answer:"

# load the model
print("loading model ...")
llm = Llama(model_path=MODEL_PATH)
print("model loaded.")

# Run the model in batch mode
print("running model ...")
output = llm(
    Prompt,
    max_tokens = 400,
    temperature=0.8,
    stop=["\n", "Question:", "Q:"],
    echo=True
    )

print(json.dumps(output, indent=2))

# Run the model in streaming mode
# print("running model in streaming mode ...")
# stream = llm(
#     Prompt,
#     max_tokens = 400,
#     temperature=0.8,
#     stop=["\n", "Question:", "Q:"],
#     stream=True
#     )
# for output in stream:
#     completionFragment = copy.deepcopy(output)
#     print(completionFragment["choices"][0]["text"], end='')
# print()