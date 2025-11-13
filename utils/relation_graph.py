def add_edge(edge_set,head_id,tail_id,label,score):
    edge_dict={'head':head_id,"tail":tail_id,"label":label,"score":score}
    if edge_dict not in edge_set:
        edge_set.append(edge_dict)
    return edge_set

def add_node_edge(node2edge_dict,node_id,type,edge_id,label,score):
    if node_id not in node2edge_dict:
        node2edge_dict[node_id]={}
    node_dict=node2edge_dict[node_id]
    if type not in node_dict:
        node_dict[type]=[]
    edge_info={"id":edge_id,"label":label,"score":score}
    if edge_info not in node_dict[type]:
        node_dict[type].append(edge_info)
    return node2edge_dict