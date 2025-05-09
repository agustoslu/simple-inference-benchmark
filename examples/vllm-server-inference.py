from pathlib import Path
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from llmlib.base_llm import Message


# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

video_path = "/home/tomasruiz/datasets/dss_home/toxicainment/2025-02-07-saxony-labeled-data/media/@xxagostino_video_7397013274937560352.mp4"
assert Path(video_path).exists(), video_path

video_messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is in the video?"},
            {"type": "video_url", "video_url": {"url": video_path}},
        ],
    },
]

convo1 = [
    Message(role="system", msg="You are a helpful assistant."),
    Message(role="user", msg="What is in the video?", video=Path(video_path)),
]

img_path = "/home/tomasruiz/datasets/dss_home/toxicainment/2025-02-07-saxony-labeled-data/media/@_de__olli__img_7414150286517783840_1.jpg"
assert Path(img_path).exists(), img_path

img_messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is in the image?"},
            {"type": "image_url", "image_url": {"url": img_path}},
        ],
    },
]

convo2 = [
    Message(role="system", msg="You are a helpful assistant."),
    Message(role="user", msg="What is in the image?", img=Path(img_path)),
]


model_id = "Qwen/Qwen2.5-VL-3B-Instruct"


def generate(client: OpenAI, messages: list[dict]):
    completion: ChatCompletion = client.chat.completions.create(
        model=model_id,
        messages=messages,
    )
    return completion


chat_response = generate(client, video_messages)
print("Chat response:", chat_response.choices[0].message.content)

chat_response = generate(client, img_messages)
print("Chat response:", chat_response.choices[0].message.content)
