"""
消息格式转换器：将自定义格式转换为 OpenAI 兼容的 messages 格式
"""

from typing import List, Dict, Any, Optional
import json


def convert_to_openai_messages(raw_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    将自定义格式的消息列表转换为 OpenAI 兼容的 messages 格式
    
    Args:
        raw_messages: 原始消息列表，包含各种类型的消息和工具调用
        
    Returns:
        List[Dict[str, Any]]: OpenAI 兼容的 messages 格式
    """
    openai_messages = []
    pending_tool_calls = []
    current_assistant_message = None
    
    for msg in raw_messages:
        msg_type = msg.get('type')
        
        if msg_type == 'message' or ('role' in msg and msg_type is None):
            # 处理普通消息（包括没有 type 字段的消息）
            role = msg.get('role')
            content = msg.get('content')
            
            # 如果有待处理的工具调用，先添加到当前助手消息中
            if pending_tool_calls and current_assistant_message:
                current_assistant_message['tool_calls'] = pending_tool_calls
                openai_messages.append(current_assistant_message)
                pending_tool_calls = []
                current_assistant_message = None
            
            if role == 'user':
                openai_messages.append({
                    "role": "user",
                    "content": content
                })
            elif role == 'assistant':
                # 处理助手消息内容
                if isinstance(content, list):
                    # 提取文本内容
                    text_content = ""
                    for item in content:
                        if item.get('type') == 'output_text':
                            text_content += item.get('text', '')
                    content = text_content
                
                current_assistant_message = {
                    "role": "assistant",
                    "content": content
                }
                
        elif msg_type == 'function_call':
            # 处理工具调用
            tool_call = {
                "id": msg.get('call_id', msg.get('id', 'unknown')),
                "type": "function",
                "function": {
                    "name": msg.get('name'),
                    "arguments": msg.get('arguments')
                }
            }
            pending_tool_calls.append(tool_call)
            
        elif msg_type == 'function_call_output':
            # 处理工具调用结果
            # 先添加带工具调用的助手消息
            if pending_tool_calls and current_assistant_message:
                current_assistant_message['tool_calls'] = pending_tool_calls
                openai_messages.append(current_assistant_message)
                current_assistant_message = None
                
                # 添加工具响应消息
                for tool_call in pending_tool_calls:
                    if tool_call['id'] == msg.get('call_id'):
                        openai_messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call['id'],
                            "content": msg.get('output', '')
                        })
                
                pending_tool_calls = []
    
    # 处理最后剩余的消息
    if current_assistant_message:
        if pending_tool_calls:
            current_assistant_message['tool_calls'] = pending_tool_calls
        openai_messages.append(current_assistant_message)
    
    return openai_messages


def convert_single_conversation(conversation_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    转换单个对话的消息格式
    
    Args:
        conversation_data: 包含完整对话的消息列表
        
    Returns:
        List[Dict[str, Any]]: OpenAI 格式的消息列表
    """
    return convert_to_openai_messages(conversation_data)


# 示例使用
if __name__ == "__main__":
    # 测试数据
    test_data = [
        {
            'content': 'You are a tool executor that completes the task assigned by the planner. Try you best to generate a precise command to call the candidate tool(s) to solve the task.\n\nTask: Search the web for current weather conditions in Shanghai to obtain real-time temperature, conditions, and other relevant details.\nContext: The query is "What is the weather in Shanghai now?". No previous steps have been executed, so no additional context is needed.\n\nInstructions:\n1. Analyze the tool required parameters from its metadata.\n2. Choose the promising tool(s) that could address the task using the provided context.\n2. Construct valid tool call format to ensure the tool name and parameters can be parsed correctly.\n3. You must at least perform one tool call.',
            'role': 'user'
        },
        {
            'id': '__fake_id__',
            'content': [
                {
                    'annotations': [],
                    'text': "I'll search the web for current weather conditions in Shanghai to get real-time temperature, conditions, and other relevant details.",
                    'type': 'output_text'
                }
            ],
            'role': 'assistant',
            'status': 'completed',
            'type': 'message'
        },
        {
            'arguments': '{"query": "current weather Shanghai temperature conditions real-time", "num_results": 5}',
            'call_id': 'call_00_C9tK6ak93TPebl3S9J7SoUFl',
            'name': 'search_web',
            'type': 'function_call',
            'id': '__fake_id__'
        },
        {
            'call_id': 'call_00_C9tK6ak93TPebl3S9J7SoUFl',
            'output': '1. Weather for Shanghai , Shanghai Municipality, China (https://www.timeanddate.com/weather/china/shanghai)\nbody: Current weather in Shanghai and forecast for today, tomorrow, and next 14 days.\n2. Weather in Shanghai — Weather forecast in Shanghai , China (https://meteum.ai/weather/en/shanghai)\nbody: Shanghai , current weather : clear. No precipitation expected today. Air temperature +12°, feels like +9°. Wind speed 3 Meters per second, northerly. Pressure 764 millimeters of mercury. Humidity 72%.\n3. Shanghai , Shanghai , China Hourly Weather | AccuWeather (https://www.accuweather.com/en/cn/shanghai/106577/hourly-weather-forecast/106577)\nbody: Hourly weather forecast in Shanghai , Shanghai , China. Check current conditions in Shanghai , Shanghai , China with radar, hourly, and more.\n4. Weather - Shanghai - 14-Day Forecast & Rain | Ventusky (https://www.ventusky.com/shanghai)\nbody: Shanghai - Weather forecast for 14 days, information from meteorological stations, webcams, sunrise and sunset, wind and precipitation maps for this place.\n5. 14-day weather forecast for Shanghai . - BBC Weather (https://www.bbc.com/weather/1796236)\nbody: Shanghai - Weather warnings issued. 14-day forecast. Add to your locationsAdd to your locations.',
            'type': 'function_call_output'
        }
    ]
    
    # 转换格式
    openai_messages = convert_to_openai_messages(test_data)
    
    # 打印结果
    print("转换后的 OpenAI 格式消息:")
    print(json.dumps(openai_messages, indent=2, ensure_ascii=False))
    
    print("\n" + "="*50)
    print("预期的消息结构:")
    for i, msg in enumerate(openai_messages):
        print(f"{i+1}. Role: {msg['role']}")
        if 'tool_calls' in msg:
            print(f"   Content: {msg['content']}")
            print(f"   Tool calls: {len(msg['tool_calls'])} 个")
        elif msg['role'] == 'tool':
            print(f"   Tool call ID: {msg['tool_call_id']}")
            print(f"   Content length: {len(msg['content'])} 字符")
        else:
            print(f"   Content: {msg['content'][:100]}...")