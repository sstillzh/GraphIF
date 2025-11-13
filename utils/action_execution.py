
def get_context_anchored(chat_vllm,user_instruction,conv_history,user_intention,action_reason,j,candidate_id):
    prompt_dir='path/GraphIF/prompts/identify_anchored.txt'
    with open(prompt_dir,'r') as f:
        prompt_template=f.read()
    system_prompt,user_template=prompt_template.split("<system>")
    user_instruction+='\n(Current Instruction: Round {})'.format(j)
    conv_history="The following is a summary of the historical conversation. The complete conversation content will be used when actually replying.\n"+conv_history
    user_prompt=(
        user_template.replace('{user_instruction}',user_instruction)
        .replace("{conv_history}",conv_history)
        .replace('{intention}',user_intention)
        .replace("{reason_context_anchored}",action_reason)
        .replace("{candidate_id}",str(candidate_id))
    )

    message=[
        {"role":"system","content":system_prompt},
        {"role":"user","content":user_prompt}
    ]
    resp=chat_vllm.chat_with_vllm(message)
    
    return resp

def get_context_anchored_update(chat_vllm,user_instruction,conv_history,user_intention,notebook,action_reason,j,id_list,candidate_id):
    prompt_dir='path/GraphIF/prompts/identify_anchored_update.txt'
    with open(prompt_dir,'r') as f:
        prompt_template=f.read()
    system_prompt,user_template=prompt_template.split("<system>")
    user_instruction+='\n(Current Instruction: Round {})'.format(j)
    conv_history="The following is a summary of the historical conversation. The complete conversation content will be used when actually replying.\n"+conv_history
    user_prompt=(
        user_template.replace('{user_instruction}',user_instruction)
        .replace("{conv_history}",conv_history)
        .replace('{intention}',user_intention)
        .replace("{notebook}", notebook)
        .replace("{id_list}", str(id_list))
        .replace("{candidate_id}", str(candidate_id))
        .replace("{reason_context_anchored}",action_reason)
    )

    message=[
        {"role":"system","content":system_prompt},
        {"role":"user","content":user_prompt}
    ]
    resp=chat_vllm.chat_with_vllm(message)
    return resp

def get_modify(chat_vllm,user_instruction,conv_history,user_intention,action_reason,j,candidate_id):
    prompt_dir='path/GraphIF/prompts/identify_modify.txt'
    with open(prompt_dir,'r') as f:
        prompt_template=f.read()
    system_prompt,user_template=prompt_template.split("<system>")
    user_instruction+='\n(Current Instruction: Round {})'.format(j)

    conv_history="The following is a summary of the historical conversation. The complete conversation content will be used when actually replying.\n"+conv_history
    user_prompt=(
        user_template.replace('{user_instruction}',user_instruction)
        .replace("{conv_history}",conv_history)
        .replace('{intention}',user_intention)
        .replace("{reason_modify}",action_reason)
        .replace("{candidate_id}",str(candidate_id))
    )

    message=[
        {"role":"system","content":system_prompt},
        {"role":"user","content":user_prompt}
    ]
    resp=chat_vllm.chat_with_vllm(message)
    
    return resp

def get_modify_update(chat_vllm,user_instruction,conv_history,user_intention,notebook,action_reason,j,id_list,candidate_id):
    prompt_dir='path/GraphIF/prompts/identify_modify_update.txt'
    with open(prompt_dir,'r') as f:
        prompt_template=f.read()
    system_prompt,user_template=prompt_template.split("<system>")
    user_instruction+='\n(Current Instruction: Round {})'.format(j)
    conv_history="The following is a summary of the historical conversation. The complete conversation content will be used when actually replying.\n"+conv_history
    user_prompt=(
        user_template.replace('{user_instruction}',user_instruction)
        .replace("{conv_history}",conv_history)
        .replace('{intention}',user_intention)
        .replace("{notebook}",notebook)
        .replace("{reason_modify}",action_reason)
        .replace("{id_list}",str(id_list))
        .replace("{candidate_id}",str(candidate_id))
    )

    message=[
        {"role":"system","content":system_prompt},
        {"role":"user","content":user_prompt}
    ]
    resp=chat_vllm.chat_with_vllm(message)
    
    return resp

def get_summary(chat_vllm,user_instruction,conv_history,user_intention,summary_reason,candidate_id,j):
    prompt_dir='path/GraphIF/prompts/identify_summary.txt'
    with open(prompt_dir,'r') as f:
        prompt_template=f.read()
    system_prompt,user_template=prompt_template.split("<system>")
    user_instruction+='\n(Current Instruction: Round {})'.format(j)
    
    user_prompt=(
        user_template.replace('{user_instruction}',user_instruction)
        .replace("{conv_history}",conv_history)
        .replace('{intention}',user_intention)
        .replace("{reason_summary}",summary_reason)
        .replace("{candidate_id}",str(candidate_id))
    )
    message=[
        {"role":"system","content":system_prompt},
        {"role":"user","content":user_prompt}
    ]
    resp=chat_vllm.chat_with_vllm(message)
    
    return resp

def check_topic(chat_vllm,user_instruction,conv_history,user_intention,j):
    prompt_dir='path/GraphIF/prompts/check_new_topic.txt'
    with open(prompt_dir,'r') as f:
        prompt_template=f.read()
    system_prompt,user_template=prompt_template.split("<system>")
    user_instruction+='\n(Current Instruction: Round {})'.format(j)
    conv_history="The following is a summary of the historical conversation. The complete conversation content will be used when actually replying.\n"+conv_history
    user_prompt=(
        user_template.replace('{user_instruction}',user_instruction)
        .replace("{conv_history}",conv_history)
        .replace('{intention}',user_intention)
        
    )
    message=[
        {"role":"system","content":system_prompt},
        {"role":"user","content":user_prompt}
    ]
    resp=chat_vllm.chat_with_vllm(message)
    return resp

def choose_topic(chat_vllm,topic_set,topic_header,history_info_list,user_instruction,user_intention):
    prompt_dir='path/GraphIF/prompts/choose_topic.txt'
    with open(prompt_dir,'r')as f:
        prompt=f.read()
    system_prompt,user_template=prompt.split('<system>')
    conv_history_of_each_topic=""
    for i,topic_id in topic_set.items():
        if i==topic_header:
            continue
        conv_history_of_each_topic+=f"Topic{i}:\n"
        for conv_id in topic_id:
            conv_item=history_info_list[conv_id]
            conv_history_of_each_topic+=f"Round{conv_id}:\n  <User Instruction>:{conv_item['instruction']}\n  <Response>:{conv_item['response']}\n"
        conv_history_of_each_topic+='\n\n\n'
    user_prompt=(
        user_template.replace("{conv_history_of_each_topic}",conv_history_of_each_topic)  
        .replace("{user_instruction}",user_instruction)
        .replace("{user_intention}",user_intention)
    )
    message=[
        {"role":"system","content":system_prompt},
        {"role":"user","content":user_prompt}
    ]
    resp=chat_vllm.chat_with_vllm(message)
    return resp

def get_notebook(history_info_list,recollection_set,anchored_set,modify_set,summary_set):
    if len(recollection_set)==0 and len(anchored_set)==0 and len(modify_set)==0 and len(summary_set)==0:
        notebook='No relevant dialogue has been extracted yet'
        return notebook
    notebook='Related content that has been identified:\n'
    recollection_str=anchored_str=modify_str=summary_str=''
    if recollection_set!=[]:
        recollection_str='(Recollection Relationship)The constraints that the response needs to meet are :\n '
        for conv_id in recollection_set:
            if conv_id>=len(history_info_list):
                break
            conv_item=history_info_list[conv_id]
            recollection_str+="--Round{}:\n  {}\n".format(conv_id,conv_item['conv_summary'])
    if anchored_set!=[]:
        anchored_str='(Context_Anchored Relationship)The context on which the instruction is based :'
        for conv_id in anchored_set:
            conv_item=history_info_list[conv_id]
            anchored_str+="--Round{}:\n  {}\n".format(conv_id,conv_item['conv_summary'])
    if modify_set!=[]:
        modify_str='(Modify Relationship)The current instruction semantically clarifies/modifies/refines the following dialogue content (If the following is empty, it means that no corresponding content is recognized):\n'
        for conv_id in modify_set:
            conv_item=history_info_list[conv_id]
            modify_str+="--Round{}:\n  {}\n".format(conv_id,conv_item['conv_summary'])
    if summary_set!=[]:
        summary_str='(Summary Relationship) The current instruction semantically requires to summarize the following dialogue content:\n'
        for conv_id in summary_set:
            conv_item=history_info_list[conv_id]
            summary_str+="--Round{}:\n  {}\n".format(conv_id,conv_item['conv_summary'])
    
    notebook+=f"{recollection_str}\n{anchored_str}\n{modify_str}\n{summary_str}"
    return notebook

def get_thought(action_list,notebook):
    thought='Summarize the current progress and gain insights from it.\n\nThe actions that have been executed so far:\n'
    for idx,action_item in enumerate(action_list):
        thought+=f'{idx+1}.{action_item}'
        thought+='\n'
    thought+='\n'
    thought+=notebook
    
    return thought