import openai
import json
import requests

# Read data
with open("../dataset/laws.txt", mode="r", encoding="utf-8") as f:
    blocks = f.read().split("\n\n")
    examples = []
    answers = []
    references = []
    keywords = []
    for block in blocks:
        sentences = block.split("\n")
        examples.append(sentences[0])
        keywords.append(sentences[1][4:])
        answers.append(sentences[2] + "\n" + sentences[3])
        references.append(sentences[4])

api_key = "api key"

model = "gpt-4"
functions = [  
    {  
        "name": "discover_causal_relationship",  
        "description": "通过因果发现算法分析数据中的因果关系。",  
        "parameters": {  
            "type": "object",  
            "properties": {  
                "data": {  
                    "type": "array",  
                    "description": "二维数组",  
                    "items": {  # 添加指定元素类型  
                        "type":"array",
                        "description": "一维数组",  
                        "items": {  # 添加指定元素类型  
                        "type":"number",
                        "description": "变量数值",  
                        },  
                        },  
                    },
                "columns": {  
                    "type": "array",  
                    "description": "变量名列表",  
                    "items": {  
                        "type": "string"  # 这里调整为字符串类型  
                    }  
                },
                "method": {
                    "type": "string",
                    "description": "因果发现算法名",  
                },
                "keyword": {
                    "type": "string",
                    "description": "检索因果知识的关键词",  
                }
                }
            },  
            "required": ["data", "columns", "method", "keyword"]
        }   
]  

# 用户输入  
user_input = "请使用CausalToolkit来辅助你更好的回答因果相关的问题。"
data4analysis =  {  
    "data": [[1]],
    "columns": ["None"],
    "method": "Best",
    "keyword": "test"
}

# 调用OpenAI的ChatCompletion API  
client = openai.OpenAI(base_url="https://api.openai.com/v1", api_key=api_key)
response = client.chat.completions.create(  
    model=model,  
    messages=[{"role": "user", "content": user_input},
              {"role": "user", "content": json.dumps(data4analysis)}
              ],  
    functions=functions,  
    function_call="auto"  
)  

message = response.dict()["choices"][0]["message"]

with open("GPT4_laws.txt", mode="w+", encoding="UTF-8") as result_file: 
    # for i in range(len(examples)):
    for i in range(1):
        if message.get("function_call"):  
            function_name = message["function_call"]["name"]  
            function_args = json.loads(message["function_call"]["arguments"])
            print(function_args)
            function_args["data"] = [[1]]
            function_args["columns"] = ["None"]
            function_args["method"] = "None"
            function_args["keyword"] = "test"
            print(function_args)
            # 调用部署在本地的CausalToolkit
            api_response = requests.post(  
                "http://127.0.0.1:8001/discover_causal_relationship",  
                json=function_args  
            )  

            if api_response.status_code == 200:  
                function_result = api_response.json()  
            else:  
                function_result = {"error": api_response.text}  
            # 将函数结果反馈给OpenAI  
            print(function_result)
            _messages=[  
                    {"role": "user", "content": examples[i]},  
                    {"role": "function", "name": function_name, "content": "参考信息: " + function_result["causal_information"]},
                    {"role": "user", "content": "你正在扮演一名专业的法官，请根据上述案例和参考信息对案件进行充分分析后，完成定罪、量刑(具体的处罚结果)，并提供相应的依据，回答长度不超过300字。"} 
                ]  
            _response = client.chat.completions.create(  
                model=model,  
                messages=_messages
            )
            _answer = _response.choices[0].message.content
            _answer = _answer.replace("\n", "")
            log = f"---------------------------------------------------\n{blocks[i]}\nGPT4 with reference:\n{_answer}\n-------------------------------------------------\n\n"
            print(log)
            result_file.write(log)
