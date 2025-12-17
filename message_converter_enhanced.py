"""
增强版消息格式转换器：将自定义格式转换为 OpenAI 兼容的 messages 格式
支持更多边缘情况和批量处理
"""

from typing import List, Dict, Any, Optional, Union
import json


class MessageConverter:
    """消息格式转换器类"""
    
    @staticmethod
    def extract_text_content(content: Union[str, List[Dict[str, Any]]]) -> str:
        """
        从复杂的内容结构中提取文本内容
        
        Args:
            content: 可能是字符串或包含多种类型内容的列表
            
        Returns:
            str: 提取的文本内容
        """
        if isinstance(content, str):
            return content
        
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get('type') == 'output_text':
                        text_parts.append(item.get('text', ''))
                    elif 'text' in item:
                        text_parts.append(item['text'])
                elif isinstance(item, str):
                    text_parts.append(item)
            return ''.join(text_parts)
        
        return str(content)
    
    @staticmethod
    def convert_to_openai_messages(raw_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        将自定义格式的消息列表转换为 OpenAI 兼容的 messages 格式
        
        Args:
            raw_messages: 原始消息列表
            
        Returns:
            List[Dict[str, Any]]: OpenAI 兼容的 messages 格式
        """
        openai_messages = []
        pending_tool_calls = []
        current_assistant_message = None
        tool_call_outputs = {}  # 存储工具调用输出，key 是 call_id
        
        # 第一遍：收集所有工具调用输出
        for msg in raw_messages:
            if msg.get('type') == 'function_call_output':
                call_id = msg.get('call_id')
                if call_id:
                    tool_call_outputs[call_id] = msg.get('output', '')
        
        # 第二遍：处理消息
        for msg in raw_messages:
            msg_type = msg.get('type')
            
            if msg_type == 'message' or ('role' in msg and msg_type is None):
                # 处理普通消息
                role = msg.get('role')
                content = msg.get('content')
                
                # 如果有待处理的工具调用，先添加到当前助手消息中
                if pending_tool_calls and current_assistant_message:
                    current_assistant_message['tool_calls'] = pending_tool_calls
                    openai_messages.append(current_assistant_message)
                    
                    # 添加对应的工具响应
                    for tool_call in pending_tool_calls:
                        call_id = tool_call['id']
                        if call_id in tool_call_outputs:
                            openai_messages.append({
                                "role": "tool",
                                "tool_call_id": call_id,
                                "content": tool_call_outputs[call_id]
                            })
                    
                    pending_tool_calls = []
                    current_assistant_message = None
                
                if role == 'user':
                    openai_messages.append({
                        "role": "user",
                        "content": MessageConverter.extract_text_content(content)
                    })
                elif role == 'assistant':
                    current_assistant_message = {
                        "role": "assistant",
                        "content": MessageConverter.extract_text_content(content)
                    }
                    
            elif msg_type == 'function_call':
                # 处理工具调用
                tool_call = {
                    "id": msg.get('call_id', msg.get('id', f"call_{len(pending_tool_calls)}")),
                    "type": "function",
                    "function": {
                        "name": msg.get('name', 'unknown_function'),
                        "arguments": msg.get('arguments', '{}')
                    }
                }
                pending_tool_calls.append(tool_call)
        
        # 处理最后剩余的消息
        if current_assistant_message:
            if pending_tool_calls:
                current_assistant_message['tool_calls'] = pending_tool_calls
                openai_messages.append(current_assistant_message)
                
                # 添加对应的工具响应
                for tool_call in pending_tool_calls:
                    call_id = tool_call['id']
                    if call_id in tool_call_outputs:
                        openai_messages.append({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": tool_call_outputs[call_id]
                        })
            else:
                openai_messages.append(current_assistant_message)
        
        return openai_messages
    
    @staticmethod
    def batch_convert(conversations: List[List[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
        """
        批量转换多个对话
        
        Args:
            conversations: 多个对话的列表
            
        Returns:
            List[List[Dict[str, Any]]]: 转换后的对话列表
        """
        return [MessageConverter.convert_to_openai_messages(conv) for conv in conversations]
    
    @staticmethod
    def validate_openai_format(messages: List[Dict[str, Any]]) -> bool:
        """
        验证消息是否符合 OpenAI 格式
        
        Args:
            messages: 要验证的消息列表
            
        Returns:
            bool: 是否符合格式
        """
        required_roles = {'user', 'assistant', 'tool', 'system'}
        
        for msg in messages:
            if not isinstance(msg, dict):
                return False
            
            role = msg.get('role')
            if role not in required_roles:
                return False
            
            if 'content' not in msg:
                return False
            
            # 检查工具调用格式
            if role == 'assistant' and 'tool_calls' in msg:
                tool_calls = msg['tool_calls']
                if not isinstance(tool_calls, list):
                    return False
                
                for tool_call in tool_calls:
                    if not all(key in tool_call for key in ['id', 'type', 'function']):
                        return False
                    
                    function = tool_call['function']
                    if not all(key in function for key in ['name', 'arguments']):
                        return False
            
            # 检查工具响应格式
            if role == 'tool' and 'tool_call_id' not in msg:
                return False
        
        return True


# 便捷函数
def convert_to_openai_messages(raw_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """便捷的转换函数"""
    return MessageConverter.convert_to_openai_messages(raw_messages)


# 示例和测试
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
    
    # 使用转换器
    converter = MessageConverter()
    
    # 转换格式
    openai_messages = converter.convert_to_openai_messages(test_data)
    
    # 验证格式
    is_valid = converter.validate_openai_format(openai_messages)
    
    print("转换后的 OpenAI 格式消息:")
    print(json.dumps(openai_messages, indent=2, ensure_ascii=False))
    
    print(f"\n格式验证结果: {'✅ 有效' if is_valid else '❌ 无效'}")
    
    print("\n" + "="*50)
    print("消息结构概览:")
    for i, msg in enumerate(openai_messages):
        print(f"{i+1}. Role: {msg['role']}")
        if 'tool_calls' in msg:
            print(f"   Content: {msg['content']}")
            print(f"   Tool calls: {len(msg['tool_calls'])} 个")
            for j, tool_call in enumerate(msg['tool_calls']):
                print(f"     - {j+1}. {tool_call['function']['name']} (ID: {tool_call['id']})")
        elif msg['role'] == 'tool':
            print(f"   Tool call ID: {msg['tool_call_id']}")
            print(f"   Content length: {len(msg['content'])} 字符")
        else:
            content_preview = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
            print(f"   Content: {content_preview}")