## GraphIF

Implementation of **"GraphIF: Enhancing Multi-Turn Instruction Following for Large Language Models with Relation Graph Prompt"**.


## Usage

### Environment
Install Python 3.10.0 first. For convenience, execute the following command.

``` bash
pip install -r requirements.txt
```

### Run
#### Setup Instructions for GraphIF
1. First, update the paths in `utils/action_identification.py` and `utils/action_execution.py` to true absolute paths.

2. Then, configure the parameters in `GraphIF.sh`.

3. After completing the configurations, run the following command:
``` bash
sh GraphIF.sh
```

An example of `GraphIF.sh`:
```bash
python GraphIF.py \
    --datapath  path/datasets/mteval_graphif.jsonl\
    --model_id  path/Qwen2.5-7B-Instruct\
    --save_dir  path/graphif_result.json
```
#### Running LLM Only (Without GraphIF)

To run LLM Only without GraphIF, you need to configure the parameters in `LLM_Only.sh` and then execute the following command:

``` bash
sh LLM_Only.sh
```
An example of `LLM_Only.sh`:
```bash
python LLM_Only.py \
    --datapath  path/datasets/mteval_graphif.jsonl\
    --model_id  path/Qwen2.5-7B-Instruct\
    --save_dir  path/llm_only_result.json
```

### Evaluate
Firstly, we need to configure the parameters in `evaluate/openai_api.py`, including the API key and model.
Then, configure the parameters in `evaluate.sh` and run the following command:
``` bash
sh evaluate.sh
```
An example of `evaluate.sh`:
```bash
python evaluate.py \
    --input_dir  path/graphif_result.json\
    --output_dir path/graphif_evaluate_result.json
```