import os
import json
import torch

__all__ = ["get_gpu_prop", "collect_gpu_info"]


dirname = os.path.dirname(__file__)
json_file = os.path.join(dirname, "gpu_info.json")


# 获取当前GPU的名称，容量，显存大小和核心数量
def get_gpu_prop(show=False):
    ngpus = torch.cuda.device_count()
    
    # 获取GPU属性信息
    properties = []
    for dev in range(ngpus):
        prop = torch.cuda.get_device_properties(dev)
        properties.append({
            "name": prop.name,
            "capability": [prop.major, prop.minor],
            "total_momory": round(prop.total_memory / 1073741824, 2), # unit GB
            "sm_count": prop.multi_processor_count
        })
       
    # 在控制台打印信息
    if show:
        print("cuda: {}".format(torch.cuda.is_available()))
        print("available GPU(s): {}".format(ngpus))
        for i, p in enumerate(properties):
            print("{}: {}".format(i, p))
    
    # 返回属性信息列表
    return properties


def sort(d, tmp={}):
    for k in sorted(d.keys()):
        if isinstance(d[k], dict):
            tmp[k] = {}
            sort(d[k], tmp[k])
        else:
            tmp[k] = d[k]
    return tmp


def collect_gpu_info(model_name, fps):
    fps = [round(i, 2) for i in fps]
    if os.path.exists(json_file):
        gpu_info = json.load(open(json_file))
    else:
        gpu_info = {}
    
    prop = get_gpu_prop()
    name = prop[0]["name"]
    check = [p["name"] == name for p in prop]
    if all(check):
        count = str(len(prop))
        if name in gpu_info:
            gpu_info[name]["properties"] = prop[0]
            perf = gpu_info[name]["performance"]
            if count in perf:
                if model_name in perf[count]:
                    perf[count][model_name].append(fps)
                else:
                    perf[count][model_name] = [fps]
            else:
                perf[count] = {model_name: [fps]}
        else:
            gpu_info[name] = {"properties": prop[0], "performance": {count: {model_name: [fps]}}}

        gpu_info = sort(gpu_info)
        json.dump(gpu_info, open(json_file, "w"))
    return gpu_info
    
