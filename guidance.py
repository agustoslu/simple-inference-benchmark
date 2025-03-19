# # some github repos that might be useful
# # bos_token issue / the same error message we get "UserWarning: Could not build_byte tokens from the tokenizer by encoding token strings: Round-trip encoding of tokens [Ð¾Ð] failed! Got [1456, 5691]"
# # https://github.com/guidance-ai/guidance/issues/989

# # custom chat template
# # https://github.com/guidance-ai/guidance/discussions/917

# # removing tokens
# # https://github.com/huggingface/transformers/issues/15032
# # https://github.com/QwenLM/Qwen2.5/issues/720


## current pip does not download the current version it's better to get it from their repo
# pip install --upgrade git+https://github.com/guidance-ai/guidance.git

# then we need to modify _engine.py and _model.py to run the script and reproduce the error of problematic tokens

# /workspace/toxicainment/simple-inference-benchmark/env/lib/python3.11/site-packages/guidance/models/transformers/_model.py
# /workspace/toxicainment/simple-inference-benchmark/env/lib/python3.11/site-packages/guidance/models/transformers/_engine.py
# _engine.py --> line 412 replace self.model = self.model_obj.config["_name_or_path"] with self.model = self.model_obj.config._name_or_path
# _model.py --> line 22 add model_name = getattr(model.config, "model_type", str(model)) before if re.search("Llama-3.*-Vision", model_name):

# what I have tried out is replacing keys in vocab.json that are tokens to <UNK> however this results in same <UNK> token having different index numbers
# manually doing that and with a script similar to the one that described here https://github.com/QwenLM/Qwen2.5/issues/720
# we should be careful about not overwriting vocab.json, tokenizer.json and merges.txt by doing something like tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, local_files_only=True)
# the script suggested in the github issue changes the format using L function from fastai we might need to keep the original format to prevent any corrupted files issues
# there are multiple occurences of ["ï¿½", "Ð¾"] in vocab in different word settings however replacing all of them with <UNK> could also lead to some problems

import torch
import guidance
from guidance import models, gen, select, user, system, assistant
from guidance.models import Transformers
from guidance.chat import ChatTemplate
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import json

# problematic tokens ["ï¿½", "Ð¾"]


model_id = "openbmb/MiniCPM-V-2_6"
tokenizer = AutoTokenizer.from_pretrained(
    model_id, trust_remote_code=True, force_download=True
)
model = AutoModel.from_pretrained(
    model_id,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
).eval()

# create custom chat template for minicpm
mini_template = tokenizer.chat_template


class MiniCPMChatTemplate(ChatTemplate):
    template_str = mini_template

    def get_role_start(self, role_name):
        if role_name == "system":
            return "<|im_start|>system\n"
        elif role_name == "user":
            return "<|im_start|>user\n"
        elif role_name == "assistant":
            return "<|im_start|>assistant\n"
        else:
            raise ValueError(f"Unsupported role: {role_name}")

    def get_role_end(self, role_name=None):
        return "<|im_end|>\n"


# wrapping the model with guidance
guidance_model = Transformers(
    model=model, tokenizer=tokenizer, chat_template=MiniCPMChatTemplate
)


# define the function for image classification task
@guidance
def classify_image(lm, image_path, question):
    image = Image.open(image_path).convert("RGB")
    lm += """
    {{#system~}}
    You are an AI assistant that provides JSON-formatted answers for image classification tasks.
    {{~/system}}

    {{#user~}}
    Analyze the given image and determine if it contains a robot and R2D2 from Star Wars. Return the response in the following structured JSON format:
    ```json
    {
        "answers": [
            {
                "question": "is_robot",
                "answer": "<yes|no>",
                "comment": "<brief explanation>"
            },
            {
                "question": "is_r2d2",
                "answer": "<yes|no>",
                "comment": "<brief explanation>"
            }
        ]
    }
    ```
    {{~/user}}

    {{#assistant~}}
    {{gen 'response' max_tokens=200}}
    {{~/assistant}}
    """

    # concatenate image and question
    lm += f"Image Path: {image_path}\n"
    lm += f"Question: {question}\n"

    return lm


image_path = "./r2d2.jpg"
question = "What is in the image?"

response = guidance_model + classify_image(image_path=image_path, question=question)
print(response)
