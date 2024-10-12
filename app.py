import torch
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
)
from PIL import Image
import gradio as gr
from threading import Thread
from transformers import TextIteratorStreamer

# Add the new imports
from transformers import AutoModelForCausalLM, CodeGenTokenizerFast as Tokenizer

# Add TESTING variable
TESTING = True  # You can change this to False when not testing

# Hugging Face model id
model_id = "mistral-community/pixtral-12b"

# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Modify the model and processor initialization
if TESTING:
    model_id = "vikhyatk/moondream1"
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    processor = Tokenizer.from_pretrained(model_id)
else:
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )
    processor = AutoProcessor.from_pretrained(model_id)


def bot_streaming(message, history, max_new_tokens=250):
    txt = message["text"]
    images = []

    if len(message["files"]) == 1:
        if isinstance(message["files"][0], str):  # examples
            image = Image.open(message["files"][0]).convert("RGB")
        else:  # regular input
            image = Image.open(message["files"][0]["path"]).convert("RGB")
        images.append(image)

    # Construct conversation history
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": images[0] if images else None},
                {"type": "text", "text": txt},
            ],
        }
    ]

    # Apply chat template
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    # Process inputs
    inputs = processor(text=prompt, images=images, return_tensors="pt").to("cuda")

    streamer = TextIteratorStreamer(
        processor.tokenizer, skip_special_tokens=True, skip_prompt=True
    )

    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=max_new_tokens)

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    buffer = ""

    for new_text in streamer:
        buffer += new_text
        yield buffer


demo = gr.ChatInterface(
    fn=bot_streaming,
    title="PixNutrition",
    examples=[
        [
            {
                "text": "What's in this image?",
                "files": ["./examples/mistral_breakfast.jpg"],
            },
            200,
        ],
    ],
    textbox=gr.MultimodalTextbox(),
    cache_examples=False,
    description="Chat with Pix Nutrition. Upload your meals plans and chat with a nutrition expert AI.",
    stop_btn="Stop Generation",
    fill_height=True,
    multimodal=True,
)

demo.launch(debug=True)
