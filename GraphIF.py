import json
import logging
import os
from datetime import datetime
from utils.data_process import *
from utils.action_identification import *
from utils.action_execution import *
from utils.response_rewrite import *
from utils.chat_vllm import ChatVLLM
from utils.relation_graph import *
import re
import argparse


def setup_logger(model_id):
    """
    Setup logger with timestamped filename
    """
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"graphif_{timestamp}_{model_id}.log"
    log_path = os.path.join(log_dir, log_filename)
    
    # Configure logger
    logger = logging.getLogger(f"GraphIF_{timestamp}")
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


def agent(datapath,model_id,save_dir):
    # Setup logger
    logger = setup_logger(model_id)
    logger.info(f"GraphIF agent started with model: {model_id}")
    
    data=read_jsonl(datapath)

    chat_vllm=ChatVLLM(model_id)   
    output_data=[]
    
    for i,row in enumerate(data):
        logger.info(f"Processing data item {i}")
        output_dict={}
        output_dict['conv']=[]
        output_dict['id']=i
        
        history_info_list=[]             
        # Data structure to record all 'global recollection instructions'
        global_inst_set=[]
        # Edge list
        edge_set=[]
        # Edges connected to each node: dict
        node2edge_dict={}
        system_prompt="You are a helpful, respectful and honest assistant."
        # conv_io=deepcopy(config[model_name]["chat_template"])
        # conv_io.set_system_message(system_prompt)
        io_message=[{'role':"system",'content':system_prompt}]
        

        # Topic set
        topic_set={0:[]}
        topic_header=0
       
        for j,turn in enumerate(row['conv']):
            logger.info(f"Round {j} conversation")
            user_instruction=turn['user']
            gold_response=turn['sys']
            action_list=[]
            anchored_set=[]

            modify_set=[]
            summary_set=[]
            logger.info(f"----User instruction: {user_instruction}")
            misjudge_tpoic=False
            # Connect with global nodes
            for recollection_id in global_inst_set:
                edge_set=add_edge(edge_set,j,recollection_id,"global constraint",1)
                node2edge_dict=add_node_edge(node2edge_dict,j,"head",recollection_id,"global constraint",1)
            
            if j==0:
                hist_info="There is no conversation history, the current instruction is the beginning of the conversation"
            else:
                complete_history_info=''
                for idx,hist_item in enumerate(history_info_list):
                    if idx not in topic_set[topic_header]:
                        continue
                    history_inst=hist_item['instruction'].strip()
                    history_resp=hist_item['response'].strip()
                    complete_history_info+=f"--Round{idx}:\n   <Instruction>\n{history_inst}\n   <Assistant>\n{history_resp}\n"
                hist_info=complete_history_info
            # First identify user intention
            user_intention=get_intention(chat_vllm,user_instruction)
            logger.info(f"---User intention: {user_intention}")
            # IDs of already extracted relationships
            id_list=[]
            # Whether it's an update extraction
            is_update=False
            action_answer=''
            while "done" not in action_answer:
                is_change=0
                
                if id_list!=[]:
                    is_update=True
                # Generate current notebook
                notebook=get_notebook(history_info_list,global_inst_set,anchored_set,modify_set,summary_set)
                
                #print(f"Current notebook: {notebook}",file=open(debug_dir,'a'))
                thought=get_thought(action_list,notebook)
                if id_list==[]:
                    action_resp=get_action(chat_vllm,user_instruction,user_intention,hist_info,misjudge_tpoic)
                else:
                    all_list=[]
                    for id in global_inst_set:
                        all_list.append(id)
                    for id in id_list:
                        if id not in all_list:
                            all_list.append(id)
                    
                    summary=""
                    if global_inst_set !=[]:
                        for id in global_inst_set:
                            summary+="The Global Constraint relationship is identified with Round-{}\n".format(id)
                    if anchored_set!=[]:
                        for id in anchored_set:
                            summary+="The Context-Anchored relationship is identified with Round-{}\n".format(id)
                    if modify_set!=[]:
                        for id in modify_set:
                            summary+="The Modify relationship is identified with Round-{}\n".format(id)
                    if summary_set!=[]:
                        for id in summary_set:
                            summary+="The Summary relationship is identified with Round-{}\n".format(id)
                    summary+="The relationship with rounds {} has been identified, and these rounds will not be considered in this action.\n".format(str(all_list))
                    summary+='Please consider the relationship with other conversation rounds. If the currently extracted information can be used to generate a response, execute the Done action.'
                    action_resp=get_action_update(chat_vllm,user_instruction,user_intention,thought,summary,hist_info)
                    
                logger.info(f"----{action_resp}") 
                action_resp=action_resp.lower()
                match = re.search(r'"rationale":\s*"(.*?)",\s*"score"', action_resp, re.DOTALL)
                if not match:
                    logger.warning("No matching content found")
                    action_reason=''
                else:
                    action_reason=match.group(1) 
                action_answer=action_resp.split('answer')[-1].split("}")[0]
                action_list.append(action_answer)
                logger.info(f"----Action to execute: {action_answer}") 
                candidate_id=[]
                for idx in range(j):
                    if idx not in id_list and idx in topic_set[topic_header]:
                        candidate_id.append(idx)
                # Take different operations for different actions
                if 'global'in action_answer:
                    if j not in topic_set[topic_header]:
                        topic_set[topic_header].append(j)
                    if j not in global_inst_set:
                        global_inst_set.append(j)
                        id_list.append(j)
                        is_change=1
                        logger.info("Current instruction identified as global constraint")
                    if is_change==0 :
                        logger.warning("No information update! Forced to break the loop!")
                        break    
                elif "anchor" in action_answer:
                    if j not in topic_set[topic_header]:
                        topic_set[topic_header].append(j)
                    if is_update:
                        anchored_resp=get_context_anchored_update(chat_vllm,user_instruction,hist_info,user_intention,notebook,action_reason,j,id_list,candidate_id)
                    else:
                        anchored_resp=get_context_anchored(chat_vllm,user_instruction,hist_info,user_intention,action_reason,j,candidate_id)
                    logger.info(f"----Locating context_anchored relationship: {anchored_resp}")
                    anchored_resp=anchored_resp.lower()
                   
                    match = re.search(r'"answer":\s*["]?(-?\d+)["]?', anchored_resp)
                    if not match:
                        logger.error(f"----Problem with model response when locating anchored relationship: {anchored_resp}")
                        
                    else:
                        chosen_id=int(match.group(1))    
                        if chosen_id>=0 and chosen_id<len(history_info_list):
                            if chosen_id not in id_list:
                                id_list.append(chosen_id)
                            #if chosen_id not in anchored_set:
                                is_change=1
                                anchored_set.append(chosen_id) 
                                logger.info(f"Identified context_anchored relationship with round {chosen_id} conversation")
                                match = re.search(r'"score"\s*:\s*([\d.]+)', anchored_resp)
                                if not match:
                                    logger.error(f"----Problem with model response when locating anchored relationship: {anchored_resp}")
                                    score=0.4
                                else:
                                    score = float(match.group(1))

                                # Update graph data structure
                                edge_set=add_edge(edge_set,j,chosen_id,"context_anchored",score)
                                node2edge_dict=add_node_edge(node2edge_dict,j,"head",chosen_id,"context_anchored",score) 
                                node2edge_dict=add_node_edge(node2edge_dict,chosen_id,"tail",j,"context_anchored",score)   
                            else:
                                logger.error("Problem occurred: duplicate identification of the same conversation round")     
                    if is_change==0 :
                        logger.warning("No information update! Forced to break the loop!")
                        break     
                elif  "modify" in action_answer:
                    if j not in topic_set[topic_header]:
                        topic_set[topic_header].append(j)
                    if is_update:
                        modify_resp=get_modify_update(chat_vllm,user_instruction,hist_info,user_intention,notebook,action_reason,j,id_list,candidate_id)
                    else:
                        modify_resp=get_modify(chat_vllm,user_instruction,hist_info,user_intention,action_reason,j,candidate_id)
                    logger.info(f"----Locating modify relationship: {modify_resp}")
                    modify_resp=modify_resp.lower()
                    match = re.search(r'"answer":\s*["]?(-?\d+)["]?', modify_resp)
                    if not match:
                        logger.error(f"----Problem with model response when locating modify relationship: {modify_resp}")
                    else:
                        chosen_id=int(match.group(1)) 
                        if chosen_id>=0 and chosen_id<len(history_info_list):
                            if chosen_id not in id_list:
                                id_list.append(chosen_id)
                            #if chosen_id not in modify_set:
                                is_change=1
                                modify_set.append(chosen_id)
                                logger.info(f"Identified modify relationship with round {chosen_id} conversation")
                                match = re.search(r'"score"\s*:\s*([\d.]+)', modify_resp)
                                if not match:
                                    logger.error(f"----Problem with model response when locating modify relationship: {modify_resp}")
                                    score=0.4
                                else:
                                    score = float(match.group(1))
                                # Update relationship graph: j->chosen_id
                                edge_set=add_edge(edge_set,j,chosen_id,"modify",score)
                                node2edge_dict=add_node_edge(node2edge_dict,j,'head',chosen_id,"modify",score)
                                node2edge_dict=add_node_edge(node2edge_dict,chosen_id,'tail',j,"modify",score)
                                
                            else:
                                logger.error("Problem occurred: duplicate identification of the same conversation round")
                    if is_change==0:
                        logger.warning("No information update! Forced to break the loop!")
                        break
                elif "summary" in action_answer:
                    if j not in topic_set[topic_header]:
                        topic_set[topic_header].append(j)
                    summary_resp=get_summary(chat_vllm,user_instruction,hist_info,user_intention,action_reason,candidate_id,j)
                    logger.info(f"----Locating summary relationship: {summary_resp}")
                    summary_resp=summary_resp.lower()
                    pattern = r'"answer":\s*\[([\d,\s]+)\]'
                    match = re.search(pattern, summary_resp)
                    if not match:
                        logger.error(f"----Problem with model response when locating summary relationship: {summary_resp}")
                    else:
                        numbers_str = match.group(1)
                        for x in numbers_str.split(','):
                            try:
                                summary_id=int(x.strip())
                            except:
                                logger.error(f"----Problem with model response when locating summary relationship: {summary_resp}")
                                continue
                            if summary_id>=0 and summary_id<len(history_info_list):
                                summary_set.append(summary_id)
                                if summary_id not in id_list:
                                    id_list.append(summary_id)
                                is_change=1
                                match = re.search(r'"score"\s*:\s*([\d.]+)', summary_resp)
                                if not match:
                                    logger.error(f"----Problem with model response when locating summary relationship: {summary_resp}")
                                    score=0.4
                                else:
                                    score = float(match.group(1))
                                logger.info(f"Identified summary relationship with round {summary_id} conversation")
                                edge_set=add_edge(edge_set,j,summary_id,"summary",score)
                                node2edge_dict=add_node_edge(node2edge_dict,j,'head',summary_id,"summary",score)
                                node2edge_dict=add_node_edge(node2edge_dict,summary_id,'tail',j,"summary",score)
                    if is_change==0:
                        logger.warning("No information update! Forced to break the loop!")
                        break
                elif "topic" in action_answer:
                    logger.info("Preliminary identification of new topic")
                    check_resp=check_topic(chat_vllm,user_instruction,hist_info,user_intention,j)
                    check_resp=check_resp.lower()
                    check_resp=check_resp.split("answer")[-1]
                    if 'yes' in check_resp:
                        logger.info("After confirmation, still considered as new topic")
                        
                        if len(topic_set.keys())>1:
                            choose_topic_resp=choose_topic(chat_vllm,topic_set,topic_header,history_info_list,user_instruction,user_intention)
                            match = re.search(r'"answer":\s*["]?(-?\d+)["]?', choose_topic_resp)
                            if not match:
                                logger.error(f"----Problem with model response when choosing topic: {choose_topic_resp}")
                            else:
                                chosen_id=int(match.group(1)) 
                                logger.info(f"Jump to topic: {chosen_id}")
                                if chosen_id>=0:
                                    if chosen_id <len(topic_set.keys()):
                                        topic_header=chosen_id
                                    else:
                                        topic_header+=1
                                        topic_set[topic_header]=[j]
                                else:
                                    topic_header+=1
                                    topic_set[topic_header]=[j]
                        else:
                            topic_header+=1
                            topic_set[topic_header]=[j]
                        break
                    else:
                        logger.info("After confirmation check, corrected - not a new topic")
                        misjudge_tpoic=True

                else:
                    #Done
                    logger.info("Identified Done, meaning no more relationships can be extracted")
                    break
            # Graph prompt module
            io_message.append({"role":"user",'content':user_instruction})
            initial_response=chat_vllm.chat_with_vllm(io_message)
            
            # Generate complete historical conversation information here
            complete_history_info=''
            for idx,hist_item in enumerate(history_info_list):
                if idx not in topic_set[topic_header]:
                    continue
                history_inst=hist_item['instruction'].strip()
                history_resp=hist_item['response'].strip()
                complete_history_info+=f"--Round{idx}:\n   <Instruction>:{history_inst}\n   <Response>:{history_resp}\n\n"
            complete_history_info+=f"--Round{j}:\n   <Instruction>:{user_instruction}\n"

            post_resp=postprocess_resp(chat_vllm,history_info_list,global_inst_set,anchored_set,modify_set,summary_set,user_instruction,user_intention,initial_response)
            parts = re.split(r'final-response', post_resp, flags=re.IGNORECASE)
            if len(parts) > 1:
                post_response = parts[-1].strip()  
            else:
                logger.warning("'final response' not found (case insensitive)")
                post_response=initial_response

            io_message.append({"role":"assistant",'content':post_response})
            conv_str=f"Instruction:{user_instruction}\nResponse:{post_response}"
            conv_summary=get_summary_dialogue(chat_vllm,conv_str)
            dialoag_ins={"instruction":turn['user'],"response":post_response,'conv_summary':conv_summary}
            history_info_list.append(dialoag_ins) 

     
            output_dict['conv'].append({"id":j,"user":user_instruction,"sys":post_response,"groundtruth":turn['sys'],"check_list":turn['check_list'],"constraint_type":turn['constraint_type']}) 
        output_data.append(output_dict)
        with open(save_dir,'w',encoding='utf-8')as f:
            json.dump(output_data,f,ensure_ascii=False,indent=4)    

    total_call=chat_vllm.output_call()
    logger.info(f"Total model calls: {total_call}")

if __name__=="__main__":
    pass
    parser = argparse.ArgumentParser(description="GraphIF")

    parser.add_argument('--datapath', type=str, required=True, help='Dataset file path')
    parser.add_argument('--model_id', type=str, default='qwen2.5-7b',required=True, help='Name of the large language model to use')
    parser.add_argument('--save_dir', type=str, required=True, help='Output JSON file save path')

    args = parser.parse_args()
    agent(
        datapath=args.datapath,
        model_id=args.model_id,
        save_dir=args.save_dir
    )