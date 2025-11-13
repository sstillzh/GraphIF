import os
import logging
import argparse
from datetime import datetime
from utils.data_process import *
from utils.chat_vllm import ChatVLLM
base_dir=os.path.dirname(os.path.abspath(__file__))


def setup_logger(model_id):
    """
    Setup logger with timestamped filename
    """
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"llm_only_{timestamp}_{model_id}.log"
    log_path = os.path.join(log_dir, log_filename)
    
    # Configure logger
    logger = logging.getLogger(f"LLM_Only_{timestamp}")
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplication
    logger.handlers.clear()
    
    # Create file handler
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logger initialized. Log file: {log_path}")
    return logger

def conv_io(datapath,model_id,save_dir):
    # Setup logger
    logger = setup_logger(model_id)
    logger.info(f"LLM Only baseline started with model: {model_id}")
    
    data=read_jsonl(datapath)

    chat_vllm=ChatVLLM(model_id)   
    output_data=[]
    for i,row in enumerate(data):
        logger.info(f"Processing data item {i}")
        output_dict={}
        output_dict['conv']=[]
        output_dict['id']=i
        system_prompt="You are a helpful, respectful and honest assistant."
        io_message=[{'role':"system",'content':system_prompt}]
        for j,turn in enumerate(row['conv']):
            logger.info(f"Round {j} conversation")
            user_instruction=turn['user']
            gold_response=turn['sys']
            logger.info(f"----User instruction: {user_instruction}")
            io_message.append({"role":"user",'content':user_instruction})
            io_response=chat_vllm.chat_with_vllm(io_message)
            logger.info(f"Response: {io_response}")
            io_message.append({"role":"assistant",'content':io_response})
            output_dict['conv'].append({"id":j,"user":user_instruction,"sys":io_response,"groundtruth":gold_response,"check_list":turn['check_list'],"constraint_type":turn['constraint_type']})
        output_data.append(output_dict)
        with open(save_dir,'w',encoding='utf-8')as f:
            json.dump(output_data,f,ensure_ascii=False,indent=4) 

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="LLM Only Baseline")
    
    parser.add_argument('--datapath', type=str, required=True, help='Dataset file path')
    parser.add_argument('--model_id', type=str, required=True, help='Name of the large language model to use')
    parser.add_argument('--save_dir', type=str, required=True, help='Output JSON file save path')
    
    args = parser.parse_args()
    conv_io(
        datapath=args.datapath,
        model_id=args.model_id,
        save_dir=args.save_dir
    )
    