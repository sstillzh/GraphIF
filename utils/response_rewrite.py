
def postprocess_resp(chat_vllm,history_info_list,recollection_set,anchored_set,modify_set,summary_set,user_instruction,user_intention,io_response):
    prompt_dir='path/GraphIF/prompts/postprocess_resp.txt'
    with open(prompt_dir,'r')as f:
        prompt=f.read()
    system_prompt,user_template=prompt.split('<system>')
    
    if len(recollection_set)==0 and len(anchored_set)==0 and len(modify_set)==0:
        notebook='No relevant dialogue has been extracted yet'
    else:    
        notebook='Related content that has been identified:\n'
        
        if anchored_set!=[]:
            anchored_str='(Context_Anchored Relationship) Context_Anchored means that :1.the current instruction semantically relies on, directly utilizes, or logically connects to specific content from the specific historical round (either user instruction or assistant response)  2.Includes both overt citations (e.g., "as mentioned in Round 2") and covert dependencies (e.g., "based on that content")\nTo answer the user instruction correctly,we need to understand the context-anchored content first.\nThe context on which the instruction is based:\n'
            for conv_id in anchored_set:
                conv_item=history_info_list[conv_id]
                anchored_str+="--Round{}:\n  <User Instruction>:{}\n  <Response>:{}\n".format(conv_id,conv_item['instruction'],conv_item['response'])
            notebook+=anchored_str
            notebook+='\n'
        if modify_set!=[]:
            modify_str='(Modify Relationship)*Modify* refers to user instructions that further develop, revise, refine, clarify, or follow up on previous content in a logical manner. It can also refer to new instructions that, while lacking an explicit contextual anchor, clearly build upon the prior conversation. In such cases, the instruction is inherently a continuation or extension of the dialogue, and thus qualifies as a Modify relationship. To respond accurately, the system must identify the part of the previous dialogue that aligns with this Modify relationship. Modify represents a more advanced form of Context-Anchored, with the key distinction being that it introduces additional or more specific requirements\nModify includes explicit modifications (e.g., "revise the previous answer") and implicit developments (e.g., "let me add another perspective")  \nThe current instruction semantically clarifies/modifies/refines the following dialogue content:\n'
            for conv_id in modify_set:
                conv_item=history_info_list[conv_id]
                modify_str+="--Round{}:\n  <User Instruction>:{}\n  <Response>:{}\n".format(conv_id,conv_item['instruction'],conv_item['response'])
            notebook+=modify_str
            notebook+='\n'
        if summary_set!=[]:
            summary_str='(Summary Relationship) The current instruction semantically requires to summarize the following dialogue content.If the INITIAL-RESPONSE does not summarize the following content well, you need to understand the following conversation content in depth and regenerate the summary:\n'
            for conv_id in summary_set:
                conv_item=history_info_list[conv_id]
                summary_str+="--Round{}:\n  <User Instruction>:{}\n  <Response>:{}\n".format(conv_id,conv_item['instruction'],conv_item['response'])
            notebook+=summary_str
            notebook+='\n'
        if recollection_set!=[]:
            recollection_str='(Global Constraint)The constraints that the response needs to meet are :\n '
            for conv_id in recollection_set:
                if conv_id>len(history_info_list):
                    break
                if conv_id==len(history_info_list):
                    recollection_str+="<User Instruction>:{user_instruction}\n".format(user_instruction=user_instruction)
                else:
                    conv_item=history_info_list[conv_id]
                    recollection_str+="<User Instruction>:{user_instruction}\n".format(user_instruction=conv_item['instruction'])
            recollection_str+='(If there are multiple Global Constraints, and the check finds that there are semantic or logical conflicts between different Global Constraints, that is, they cannot be satisfied at the same time, then the Global Constraint that appear later in the listed order will be satisfied first.)\nIf the check finds that INITIAL-REPONSE does not meet the above constraints, regenerate the response based on INITIAL-RESPONSE.\n'
            notebook+=recollection_str
            notebook+='\n'
    
    user_prompt=(
                    user_template.replace("{relevant_info}",notebook)
                    .replace("{user_instruction}",user_instruction)
                    .replace("{user_intention}",user_intention)
                    .replace("{initial_response}", io_response)
                )
    message=[
        {"role":"system","content":system_prompt},
        {"role":"user","content":user_prompt}
    ]
    resp=chat_vllm.chat_with_vllm(message)
    return resp