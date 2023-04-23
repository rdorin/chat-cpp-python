
import json, os
from llama_cpp import Llama


# Local models
Models = {
    "VICUNA 13B": "/Users/randalldorin/Developer/ggml-models/ggml-vicuna-13b-4bit-rev1.bin",
    "VICUNA 7B": "/Users/randalldorin/Developer/ggml-models/ggml-vicuna-7b-1.1-q4_0.bin",
    "ALPACA 7B": "/Users/randalldorin/Developer/ggml-models/ggml-alpaca-7b-q4.bin",
    "ALPACA 30B": "/Users/randalldorin/Developer/ggml-models/ggml-alpaca-lora-30B-q4_1.bin",
    "LLAMA 7B": "/Users/randalldorin/Developer/ggml-models/ggml-llama-7b-q4_0.bin",
    "LLAMA 13B": "/Users/randalldorin/Developer/ggml-models/ggml-llama-13b-q4_0.bin",
    "KOALA 13B": "/Users/randalldorin/Developer/ggml-models/ggml-koala-13b-123g-q4.bin",
    "OPEN ASSISTANT 13B": "/Users/randalldorin/Developer/ggml-models/ggml-oasst-llama-13b-q4.bin",
    "OPEN ASSISTANT 30B": "/Users/randalldorin/Developer/ggml-models/ggml-oasst-llama-30b-q4.bin",
}

# build the model menu for the user
model_menu = {}
model_menu = {str(i + 1): {"name": key, "path": Models[key]} for i, key in enumerate(Models)}

# prompt the user for a model
for key, option in model_menu.items():
    print(f"{key}: {option['name']}")

choice = input("Choose a model: ")

if choice in model_menu:
    model_path = model_menu[choice]["path"]
else:
    model_path = Models["VICUNA_7B"]

print("loading model...")

# load the model
llm = Llama(model_path)
print("model loaded")

# initialize the speech engine
# r = sr.Recognizer()

# initialize the text to speech engine
# tts = pyttsx3.init()

# def say(text):
#     print(text)

#     # using pyttxs3 for now
#     tts.say(text)
#     tts.runAndWait()

while True:
    respond = input("Enter a message: ")

    # prompt = "Hello, and welcome to our scientific chatbot. Our goal is to provide accurate and helpful information about a wide range of scientific topics.\n"

    prompt = "Question: {} Answer: ".format(respond)

    if respond == "quit":
        break
    
    print("\nlet me think\n")

    output = llm(
        prompt,
        max_tokens = 400,
        temperature=0.8,
        stop=["\n", "Question:", "Q:"],
        echo=False
    )

    print(json.dumps(output, indent=2))

    ai_response = output["choices"][0]["text"]
    print(ai_response)
    os.system(f"say {ai_response}")
