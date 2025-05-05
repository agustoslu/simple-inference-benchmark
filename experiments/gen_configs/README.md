# Generation Setting
In the folder there are fine-tuned generation configuration files from models' HF repositories. 
For self-consistency experiment with temp=0.7 across 5 runs, we removed paremeters that are "for manipulation of the model output logits" (https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig.temperature) and only kept the rest of it. 
For getting deterministic generations, we didn't include temperature parameter and set to do_sample to False, which then uses greedy decoding and skips default temp=1.0. [Paremeters that control the generation strategy used](https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig.do_sample)
