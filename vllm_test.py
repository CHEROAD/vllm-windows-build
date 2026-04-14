from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
import os

# 1. 初始化模型
llm = ChatOpenAI(
            model="Qwen3.5-0.8B",
            base_url="http://127.0.0.1:8100/v1",
            api_key="YOUR_API_KEY",
            temperature=0.7,
            streaming=True
        )

# 2. 定义工具
@tool
def get_weather(city: str) -> str:
    """获取城市天气信息"""
    weather_db = {
        "北京": "晴天，15-25度",
        "上海": "多云，18-28度",
        "深圳": "小雨，20-30度",
    }
    return weather_db.get(city, f"{city}的天气信息暂不可用")

@tool
def calculator(expression: str) -> str:
    """执行数学计算"""
    try:
        result = eval(expression)
        return f"计算结果: {expression} = {result}"
    except:
        return "计算错误"

@tool  
def search_knowledge(query: str) -> str:
    """搜索知识库"""
    knowledge = {
        "LangChain": "LangChain是一个用于开发LLM应用的框架，支持工具、代理、内存管理等功能。",
        "机器学习": "机器学习是AI的子集，让系统能从数据中自动学习和改进。",
    }
    for key, value in knowledge.items():
        if key in query:
            return value
    return f"未找到关于'{query}'的信息"

# 3. 创建 Agent
agent = create_agent(
    model=llm,
    tools=[get_weather, calculator, search_knowledge],
    system_prompt="你是一个专业的中文助手。仔细分析用户问题，选择合适的工具来回答。"
)

# 4. 使用 Agent
result = agent.invoke({
    "messages": [{"role": "user", "content": "北京今天天气怎么样？"}]
})

# 获取回答
final_message = result["messages"][-1]
print(f"回答: {final_message.content}")
