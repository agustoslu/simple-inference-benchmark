import math
import numpy as np
from PIL import Image
from moviepy import VideoFileClip
import tempfile
import librosa
import soundfile as sf

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

# load omni model default, the default init_vision/init_audio/init_tts is True
# if load vision-only model, please set init_audio=False and init_tts=False
# if load audio-only model, please set init_vision=False
model_id = 'openbmb/MiniCPM-o-2_6'

def load_model():
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        attn_implementation='flash_attention_2', # sdpa or flash_attention_2
        torch_dtype=torch.bfloat16,
        init_vision=True,
        init_audio=True,
        init_tts=False
    )
    return model.eval().to("cuda")

def load_tokenizer(model_id=model_id):
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


def get_video_chunk_content(video_path, flatten=True):
    """Returns a list of interleaved '<unit>', image, and audio chunks."""
    video = VideoFileClip(video_path)
    print('video_duration:', video.duration)
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio_file:
        temp_audio_file_path = temp_audio_file.name
        video.audio.write_audiofile(temp_audio_file_path, codec="pcm_s16le", fps=16000)
        audio_np, sr = librosa.load(temp_audio_file_path, sr=16000, mono=True)
    num_units = math.ceil(video.duration)
    
    # 1 frame + 1s audio chunk
    contents= []
    for i in range(num_units):
        frame = video.get_frame(i+1)
        image = Image.fromarray((frame).astype(np.uint8))
        audio = audio_np[sr*i:sr*(i+1)]
        if flatten:
            contents.extend(["<unit>", image, audio])
        else:
            contents.append(["<unit>", image, audio])
    
    return contents


def generate(model, tokenizer, question: str, video_path: str) -> str:
    sys_msg = model.get_sys_prompt(mode='omni', language='en')
    question_msg = {"role": "user", "content": [question]}

    contents = get_video_chunk_content(video_path)
    print(f"video {video_path} split into {len(contents) / 3} chunks")

    content_msg = {"role":"user", "content": contents}
    msgs = [sys_msg, content_msg, question_msg]

    res = model.chat(
        msgs=msgs,
        tokenizer=tokenizer,
        sampling=True,
        temperature=0.5,
        max_new_tokens=4096,
        omni_input=True,
        use_tts_template=True,
        generate_audio=False,
        max_slice_nums=1,
        use_image_id=False,
        return_dict=True
    )
    return res.text

if __name__ == "__main__":
    model = load_model()
    tokenizer = load_tokenizer()
    # In addition to vision-only mode, tts processor and vocos also needs to be initialized
    # model.init_tts()
    video_path="assets/Skiing.mp4"
    question = "What color is the backpack?"
    res = generate(model, tokenizer, question, video_path)
    print(res)
