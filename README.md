# simple-inference-benchmark
Benchmarking different runtimes for vision language models


| Model                   | Runtime     | QPS@1 | TPS@1 | TPQ@1 | VRAMpeak@1 |
| ----------------------- | ----------- | ----- | ----- | ----- | ---------- |
| MiniCPM                 | huggingface | ?     | ?     | ?     | ?          |
| MiniCPM (model.compile) | huggingface | ?     | ?     | ?     | ?          |

# Run the experiment
download the dataset
```bash
python download.py
```

Run the experiment
```bash
python run_minicpm.py
```

# Install llmlib
You want to have both codebases side by side for development purposes, so that you can introduce changes to `llmlib` and immediately test them in this project.
```bash
git clone https://github.com/tomasruizt/llm_app
cd llm_app/llmlib
pip install -e .
```

# Troubleshooting

#### Error: Gemma3 not found
```shell
ImportError: cannot import name 'Gemma3ForConditionalGeneration' from 'transformers'
```
This happend because the `transformers` version is too early and does not contain the Gemma3 model yet. Install this version:
```shell
pip install git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3
```