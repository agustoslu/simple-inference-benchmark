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

# Troubleshooting

#### Error: Gemma3 not found
```shell
ImportError: cannot import name 'Gemma3ForConditionalGeneration' from 'transformers'
```
The error Gemma not found is because of the transformers version is too early. Install this version:
```shell
pip install git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3
```