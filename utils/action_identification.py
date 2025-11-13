def get_intention(chat_vllm,user_instruction):
    prompt_dir='path/GraphIF/prompts/agent_intention.txt'
    with open(prompt_dir,'r')as f:
        prompt=f.read()
    system_prompt,user_template=prompt.split("<system>")
    user_prompt=(
        user_template.replace("{user_inst}",user_instruction)
    )  
  
    message=[
        {"role":"system","content":system_prompt},
        {"role":"user","content":user_prompt}
    ]
    resp=chat_vllm.chat_with_vllm(message)
    return resp

def get_summary_dialogue(chat_vllm,conv_info):
    prompt_dir='path/GraphIF/prompts/conv_summary.txt'
    with open(prompt_dir,'r')as f:
        prompt=f.read()
    system_prompt,user_template=prompt.split('<system>')
    user_prompt=(
        user_template.replace("{response}",conv_info)
    )
    message=[
        {"role":"system","content":system_prompt},
        {"role":"user","content":user_prompt}
    ]
    resp=chat_vllm.chat_with_vllm(message)
    return resp

def get_action(chat_vllm,instruction,intention,history_info,misjudge_topic):
    if misjudge_topic==True:
        prompt_dir='path/GraphIF/prompts/agent_action_alternative.txt'
    else:
        prompt_dir='path/GraphIF/prompts/agent_action.txt'
    with open(prompt_dir,'r') as f:
        prompt_template=f.read()
    system_prompt,user_template=prompt_template.split("<system>")
        
    action_user_prompt=(
        user_template.replace("{user_instruction}",instruction)
        .replace("{intention}",intention)
        .replace("{conv_history}", history_info)
    )

    message=[
        {"role":"system","content":system_prompt},
        {"role":"user","content":action_user_prompt}
    ]
    resp=chat_vllm.chat_with_vllm(message)
    
    return resp

def get_action_update(chat_vllm,instruction,intention,notebook,summary,history_info):
    prompt_dir='path/GraphIF/prompts/agent_action_update.txt'
    with open(prompt_dir,'r') as f:
        prompt_template=f.read()
    system_prompt,plan_user_template=prompt_template.split("<system>")
    action_user_prompt=(
        plan_user_template.replace("{user_instruction}",instruction)
        .replace("{intention}",intention)
        .replace("{notebook}",notebook)
        .replace("{summary}",summary)
        .replace("{conv_history}", history_info)
    )
    

    message=[
        {"role":"system","content":system_prompt},
        {"role":"user","content":action_user_prompt}
    ]
    resp=chat_vllm.chat_with_vllm(message)
    
    return resp