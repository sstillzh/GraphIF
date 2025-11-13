from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import os
from random import randint



class ChatVLLM:
    def __init__(self,model_id):
        # Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id,trust_remote_code=True)

        # Pass the default decoding hyperparameters of Qwen2.5-7B-Instruct
        # max_tokens is for the maximum length for generation.
        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.8, top_k=20,repetition_penalty=1.05, max_tokens=2048,seed=randint(0, 100000))

        # Input the model name or path. Can be GPTQ or AWQ models.
        self.llm = LLM(model=model_id,tensor_parallel_size = 1,gpu_memory_utilization=0.8,max_model_len=25000,trust_remote_code=True)
        
        self.call=0
    def chat_with_vllm(self,messages):
        self.call+=1
        tokenizer=self.tokenizer
        llm=self.llm
        sampling_params=self.sampling_params

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # generate outputs
        outputs = llm.generate([text], sampling_params)

        result=[]
        # Print the outputs.
        for output in outputs:
            generated_text = output.outputs[0].text
           
            result.append(generated_text)
        return result[0]
    
    def output_call(self):
        print(f"total LLM calls:{self.call}")
        return self.call


