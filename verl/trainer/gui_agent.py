import ast
import re
import math
from PIL import Image
import ray
import os
from io import BytesIO
import base64
import datetime
import traceback
import torch
from qwen_vl_utils import process_vision_info

try:
    from desktop_env.desktop_env import DesktopEnv
except ModuleNotFoundError:
    DesktopEnv = None




uitars_system_prompt = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
```
Thought: ...
Action: ...
```

## Action Space

click(start_box='<|box_start|>(x1,y1)<|box_end|>')
left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')
right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
hotkey(key='')
type(content='xxx') # Use escape characters \\', \\\", and \\n in content part to ensure we can parse the content in normal python string format. If you want to submit your input, use \\n at the end of content. 
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.
fail() # Use when you think the task is not feasible or cannot be completed.

## Note
- Use English in `Thought` and `Action` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{instruction}
"""

FINISH_WORD = "finished"
WAIT_WORD = "wait"
ENV_FAIL_WORD = "error_env"
CALL_USER = "call_user"

IMAGE_FACTOR = 28
MIN_PIXELS = 100 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

# 定义一个函数来解析每个 action
def _escape_backslashes_in_string_literals(s: str) -> str:
    """Escape raw backslashes inside single-quoted string literals so ast.parse doesn't fail
    on Windows-style paths like C:\\Users\\test where \\U is an invalid unicode escape."""
    import re as _re
    # Replace each single-quoted string's content: escape any \ that isn't already \\
    def fix_str(m):
        inner = m.group(1)
        # Escape lone backslashes (not already doubled)
        inner = _re.sub(r'\\(?!\\)', r'\\\\', inner)
        return "'" + inner + "'"
    return _re.sub(r"'([^']*)'", fix_str, s)


def parse_action(action_str):
    def _do_parse(s):
        node = ast.parse(s, mode='eval')
        if not isinstance(node, ast.Expression):
            raise ValueError("Not an expression")
        call = node.body
        if not isinstance(call, ast.Call):
            raise ValueError("Not a function call")
        if isinstance(call.func, ast.Name):
            func_name = call.func.id
        elif isinstance(call.func, ast.Attribute):
            func_name = call.func.attr
        else:
            func_name = None
        kwargs = {}
        for kw in call.keywords:
            key = kw.arg
            if isinstance(kw.value, ast.Constant):
                value = kw.value.value
            elif isinstance(kw.value, ast.Str):  # 兼容旧版本 Python
                value = kw.value.s
            else:
                value = None
            kwargs[key] = value
        return {'function': func_name, 'args': kwargs}

    try:
        return _do_parse(action_str)
    except SyntaxError:
        # Retry with backslashes escaped (handles Windows paths like C:\Users\test)
        try:
            return _do_parse(_escape_backslashes_in_string_literals(action_str))
        except Exception as e2:
            print(f"Failed to parse action '{action_str}' (after backslash escape): {e2}")
            return None
    except Exception as e:
        print(f"Failed to parse action '{action_str}': {e}")
        return None

def escape_single_quotes(text):
    # 匹配未转义的单引号（不匹配 \\'）
    pattern = r"(?<!\\)'"
    return re.sub(pattern, r"\\'", text)


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor

def linear_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    if width * height > max_pixels:
        """
        如果图片超过/低于像素限制，则计算一个缩放因子resize_factor，使图片的像素数缩小到等于或小于max_pixels。这个缩放因子是通过开平方根计算的，确保纵横比保持不变,这样原始的相对坐标可以不经转换直接复用
        """
        resize_factor = math.sqrt(max_pixels / (width * height))
        width, height = int(width * resize_factor), int(height * resize_factor)
    if width * height < min_pixels:
        resize_factor = math.sqrt(min_pixels / (width * height))
        width, height = math.ceil(width * resize_factor), math.ceil(height * resize_factor)

    return height, width 

def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar



def parse_action_to_structure_output(text, factor, origin_resized_height, origin_resized_width, model_type, max_pixels=16384*28*28, min_pixels=100*28*28):
    text = text.strip()
    if model_type == "qwen25vl":
        smart_resize_height, smart_resize_width = smart_resize(origin_resized_height, origin_resized_width, factor=IMAGE_FACTOR, min_pixels=min_pixels, max_pixels=max_pixels)

    # 正则表达式匹配 Action 字符串
    if text.startswith("Thought:"):
        thought_pattern = r"Thought: (.+?)(?=\s*Action:|$)"
        thought_hint = "Thought: "
    elif text.startswith("Reflection:"):
        thought_pattern = r"Reflection: (.+?)Action_Summary: (.+?)(?=\s*Action:|$)"
        thought_hint = "Reflection: "
    elif text.startswith("Action_Summary:"):
        thought_pattern = r"Action_Summary: (.+?)(?=\s*Action:|$)"
        thought_hint = "Action_Summary: "
    else:
        thought_pattern = r"Thought: (.+?)(?=\s*Action:|$)"
        thought_hint = "Thought: "
    reflection, thought = None, None
    thought_match = re.search(thought_pattern, text, re.DOTALL)
    if thought_match:
        if len(thought_match.groups()) == 1:
            thought = thought_match.group(1).strip()
        elif len(thought_match.groups()) == 2:
            thought = thought_match.group(2).strip()
            reflection = thought_match.group(1).strip()
    assert "Action:" in text
    action_str = text.split("Action:")[-1]

    tmp_all_action = action_str.split("\n\n")
    all_action = []
    for action_str in tmp_all_action:
        if "type(content" in action_str:
            # 正则表达式匹配 content 中的字符串并转义单引号
            def escape_quotes(match):
                content = match.group(1)  # 获取 content 的值
                return content

            # 使用正则表达式进行替换
            pattern = r"type\(content='(.*?)'\)"  # 匹配 type(content='...')
            content = re.sub(pattern, escape_quotes, action_str)

            # 处理字符串
            action_str = escape_single_quotes(content)
            action_str = "type(content='" + action_str + "')"
        elif "finished(content" in action_str:
            def escape_quotes(match):
                content = match.group(1)  # 获取 content 的值
                return content
            
            # 使用正则表达式进行替换
            pattern = r"finished\(content='(.*?)'\)"  # 匹配 type(content='...')
            content = re.sub(pattern, escape_quotes, action_str)

            # 处理字符串
            action_str = escape_single_quotes(content)
            action_str = "finished(content='" + action_str + "')"
            
        all_action.append(action_str)

    parsed_actions = [parse_action(action.replace("\n","\\n").lstrip()) for action in all_action]
    actions = []
    for action_instance, raw_str in zip(parsed_actions, all_action):
        if action_instance == None:
            print(f"Action can't parse: {raw_str}")
            raise ValueError(f"Action can't parse: {raw_str}") 
        action_type = action_instance["function"]
        params = action_instance["args"]

        # import pdb; pdb.set_trace()
        action_inputs = {}
        for param_name, param in params.items():
            if param == "": continue
            param = param.lstrip()  # 去掉引号和多余的空格
            # 处理start_box或者end_box参数格式 '<bbox>x1 y1 x2 y2</bbox>'
            action_inputs[param_name.strip()] = param
            
            if "start_box" in param_name or "end_box" in param_name:
                ori_box = param
                # Strip box tokens before parsing coordinates
                ori_box = ori_box.replace("<|box_start|>", "").replace("<|box_end|>", "")
                # Remove parentheses and split the string by commas
                numbers = ori_box.replace("(", "").replace(")", "").split(",")

                # Convert to float and scale by 1000
                # Qwen2.5vl output absolute coordinates, qwen2vl output relative coordinates
                if model_type == "qwen25vl":
                    float_numbers = []
                    for num_idx, num in enumerate(numbers):
                        num = float(num)
                        if (num_idx + 1) % 2 == 0:
                            float_numbers.append(float(num/smart_resize_height))
                        else:
                            float_numbers.append(float(num/smart_resize_width))
                else:
                    float_numbers = [float(num) / factor for num in numbers]

                if len(float_numbers) == 2:
                    float_numbers = [float_numbers[0], float_numbers[1], float_numbers[0], float_numbers[1]]
                action_inputs[param_name.strip()] = str(float_numbers)

        # import pdb; pdb.set_trace()
        actions.append({
            "reflection": reflection,
            "thought": thought,
            "action_type": action_type,
            "action_inputs": action_inputs,
            "text": text
        })
    return actions

def parsing_response_to_pyautogui_code(responses, image_height: int, image_width:int, input_swap:bool=True) -> str:
    '''
    将M模型的输出解析为OSWorld中的action，生成pyautogui代码字符串
    参数:
        response: 包含模型输出的字典，结构类似于：
        {
            "action_type": "hotkey",
            "action_inputs": {
                "hotkey": "v ctrl",
                "start_box": None,
                "end_box": None
            }
        }
    返回:
        生成的pyautogui代码字符串
    '''

    pyautogui_code = f"import pyautogui\nimport time\n"
    if isinstance(responses, dict):
        responses = [responses]
    for response_id, response in enumerate(responses):
        if "observation" in response:
            observation = response["observation"]
        else:
            observation = ""

        if "thought" in response:
            thought = response["thought"]
        else:
            thought = ""
        
        if response_id == 0:
            pyautogui_code += f"'''\nObservation:\n{observation}\n\nThought:\n{thought}\n'''\n"
        else:
            pyautogui_code += f"\ntime.sleep(1)\n"

        action_dict = response
        action_type = action_dict.get("action_type")
        action_inputs = action_dict.get("action_inputs", {})
        
        if action_type == "hotkey":
            # Parsing hotkey action
            if "key" in action_inputs:
                hotkey = action_inputs.get("key", "")
            else:
                hotkey = action_inputs.get("hotkey", "")

            if hotkey == "arrowleft":
                hotkey = "left"

            elif hotkey == "arrowright":
                hotkey = "right"
            
            elif hotkey == "arrowup":
                hotkey = "up"
            
            elif hotkey == "arrowdown":
                hotkey = "down"

            if hotkey:
                # Handle other hotkeys
                keys = hotkey.split()  # Split the keys by space
                convert_keys = []
                for key in keys:
                    if key == "space":
                        key = ' '
                    convert_keys.append(key)
                pyautogui_code += f"\npyautogui.hotkey({', '.join([repr(k) for k in convert_keys])})"
        
        elif action_type == "press":
            # Parsing press action
            if "key" in action_inputs:
                key_to_press = action_inputs.get("key", "")
            else:
                key_to_press = action_inputs.get("press", "")

            if key_to_press == "arrowleft":
                key_to_press = "left"
            elif key_to_press == "arrowright":
                key_to_press = "right"
            elif key_to_press == "arrowup":
                key_to_press = "up"
            elif key_to_press == "arrowdown":
                key_to_press = "down"
            elif key_to_press == "space":
                key_to_press = " "

            if key_to_press:
                # Simulate pressing a single key
                pyautogui_code += f"\npyautogui.press({repr(key_to_press)})"
            
        elif action_type == "keyup":
            key_to_up = action_inputs.get("key", "")
            pyautogui_code += f"\npyautogui.keyUp({repr(key_to_up)})"
        
        elif action_type == "keydown":
            key_to_down = action_inputs.get("key", "")
            pyautogui_code += f"\npyautogui.keyDown({repr(key_to_down)})"

        elif action_type == "type":
            # Parsing typing action using clipboard
            content = action_inputs.get("content", "")
            content = escape_single_quotes(content)
            stripped_content = content
            if content.endswith("\n") or content.endswith("\\n"):
                stripped_content = stripped_content.rstrip("\\n").rstrip("\n")
            if content:
                if input_swap:
                    pyautogui_code += f"\nimport pyperclip"
                    pyautogui_code += f"\npyperclip.copy({repr(stripped_content)})"
                    pyautogui_code += f"\npyautogui.hotkey('ctrl', 'v')"
                    pyautogui_code += f"\ntime.sleep(0.5)\n"
                    if content.endswith("\n") or content.endswith("\\n"):
                        pyautogui_code += f"\npyautogui.press('enter')"
                else:
                    pyautogui_code += f"\npyautogui.write({repr(stripped_content)}, interval=0.1)"
                    pyautogui_code += f"\ntime.sleep(0.5)\n"
                    if content.endswith("\n") or content.endswith("\\n"):
                        pyautogui_code += f"\npyautogui.press('enter')"

        
        elif action_type in ["drag", "select"]:
            # Parsing drag or select action based on start and end_boxes
            start_box = action_inputs.get("start_box")
            end_box = action_inputs.get("end_box")
            if start_box and end_box:
                x1, y1, x2, y2 = eval(start_box)  # Assuming box is in [x1, y1, x2, y2]
                sx = round(float((x1 + x2) / 2) * image_width, 3)
                sy = round(float((y1 + y2) / 2) * image_height, 3)
                x1, y1, x2, y2 = eval(end_box)  # Assuming box is in [x1, y1, x2, y2]
                ex = round(float((x1 + x2) / 2) * image_width, 3)
                ey = round(float((y1 + y2) / 2) * image_height, 3)
                pyautogui_code += (
                    f"\npyautogui.moveTo({sx}, {sy})\n"
                    f"\npyautogui.dragTo({ex}, {ey}, duration=1.0)\n"
                )

        elif action_type == "scroll":
            # Parsing scroll action
            start_box = action_inputs.get("start_box")
            if start_box:
                x1, y1, x2, y2 = eval(start_box)  # Assuming box is in [x1, y1, x2, y2]
                x = round(float((x1 + x2) / 2) * image_width, 3)
                y = round(float((y1 + y2) / 2) * image_height, 3)
                
                # # 先点对应区域，再滚动
                # pyautogui_code += f"\npyautogui.click({x}, {y}, button='left')"
            else:
                x = None
                y = None
            direction = action_inputs.get("direction", "")
            
            if x == None:
                if "up" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(5)"
                elif "down" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(-5)"
            else:
                if "up" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(5, x={x}, y={y})"
                elif "down" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(-5, x={x}, y={y})"

        elif action_type in ["click", "left_single", "left_double", "right_single", "hover"]:
            # Parsing mouse click actions
            start_box = action_inputs.get("start_box")
            if start_box is None:
                pass  # No coordinates provided, skip this action
            else:
                start_box = eval(str(start_box))
                if len(start_box) == 4:
                    x1, y1, x2, y2 = start_box  # Assuming box is in [x1, y1, x2, y2]
                elif len(start_box) == 2:
                    x1, y1 = start_box
                    x2 = x1
                    y2 = y1
                x = round(float((x1 + x2) / 2) * image_width, 3)
                y = round(float((y1 + y2) / 2) * image_height, 3)
                if action_type == "left_single" or action_type == "click":
                    pyautogui_code += f"\npyautogui.click({x}, {y}, button='left')"
                elif action_type == "left_double":
                    pyautogui_code += f"\npyautogui.doubleClick({x}, {y}, button='left')"
                elif action_type == "right_single":
                    pyautogui_code += f"\npyautogui.click({x}, {y}, button='right')"
                elif action_type == "hover":
                    pyautogui_code += f"\npyautogui.moveTo({x}, {y})"
        
        elif action_type in ["finished"]:
            pyautogui_code = f"DONE"
        
        else:
            pyautogui_code += f"\n# Unrecognized action type: {action_type}"

    return pyautogui_code


def add_box_token(input_string):
    # Step 1: Split the string into individual actions
    if "Action: " in input_string and "start_box=" in input_string:
        suffix = input_string.split("Action: ")[0] + "Action: "
        actions = input_string.split("Action: ")[1:]
        processed_actions = []
        for action in actions:
            action = action.strip()
            # Step 2: Extract coordinates (start_box or end_box) using regex
            coordinates = re.findall(r"(start_box|end_box)='\((\d+),\s*(\d+)\)'", action)
            
            updated_action = action  # Start with the original action
            for coord_type, x, y in coordinates:
                # Convert x and y to integers
                updated_action = updated_action.replace(f"{coord_type}='({x},{y})'", f"{coord_type}='<|box_start|>({x},{y})<|box_end|>'")
            processed_actions.append(updated_action)
        
        # Step 5: Reconstruct the final string
        final_string = suffix + "\n\n".join(processed_actions)
    else:
        final_string = input_string
    return final_string


from ..utils.tokenizer import get_processor, get_tokenizer

import verl.utils.torch_functional as VF
from ..models.transformers.qwen2_vl import get_rope_index

@ray.remote(num_cpus=1)
class EnvWorker():
    system_prompt = uitars_system_prompt
    
    ground_prompt = r"""Output only the coordinate of one point in your response. What element matches the following task: """

    def __init__(self, worker_idx, max_steps=15, config=None):
        if DesktopEnv is None:
            raise ModuleNotFoundError(
                "desktop_env is not installed. Required for local EnvWorker only. "
                "For remote execution, set env.remote_server_url."
            )
        self.worker_idx = worker_idx
        self.step_timeout = 60
        self.config = config

        self.tokenizer = get_tokenizer(
            config.worker.actor.model.model_path,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )
        self.processor = get_processor(
            config.worker.actor.model.model_path,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )

        self.model = 'uitars'
        print('Start to create desktop_env.')
        self.env = DesktopEnv(
            provider_name="docker", 
            action_space="pyautogui",
            screen_size=(1920, 1080),
            cache_dir=f"cache_dirs/cache_0",
            # cache_dir=f"cache_dirs/cache_{self.worker_idx%32}",
            headless=True,
            os_type="Ubuntu",
            require_a11y_tree=False
        )

        self.is_init = False
        self.is_done = False
        self.max_steps = max_steps
        # self.parser = Qwen2VLParser()

        self.action_parse_res_factor = 1000
        self.model_type = "qwen25vl"
        self.max_pixels = config.data.max_pixels
        self.min_pixels = config.data.min_pixels

        self.instruction = None
        self.task_config = None
        self.step_counter = 0
        self.history_images = []
        self.history_messages = []

        self.reset_train_tensors()

    def reset_train_tensors(self):
        # for training
        self.input_ids = torch.zeros((0,), dtype=torch.int64)
        self.labels = torch.full((0,), -100, dtype=torch.int64)
        self.attention_mask = torch.zeros((0, ), dtype=torch.int64)

        self.pixel_values = None
        self.image_grid_thw = None

    def load_content(self, content):
        if isinstance(content, str):
            return content
        
        if isinstance(content, list):
            return ''.join([self.load_content(c) for c in content])

        if isinstance(content, dict):
            if "text" in content:
                return content["text"]
            elif "image" in content:
                return "<|vision_start|><|image_pad|><|vision_end|>"
        
        raise ValueError(f"Unknown content type: {content}")
    
    def process_message(self, message):
        tokenizer = self.tokenizer
        processor = self.processor
        
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            message, return_video_kwargs=True)

        input_ids = []
        labels = []
        attention_mask = []

        image_count = 0
        pixel_values = []
        image_grid_thw = []
        for turn_idx, msg in enumerate(message):
            role = msg['role']
            content = self.load_content(msg['content'])
            prompt = f'<|im_start|>{role}\n' + content + '<|im_end|>\n'

            cur_image_num = prompt.count("<|image_pad|>")                
            if cur_image_num > 0:
                result = processor(image_inputs[image_count:image_count+cur_image_num], [prompt], add_special_tokens=False, return_tensors="pt")
                image_count += cur_image_num
            else:
                result = processor(None, [prompt], add_special_tokens=False, return_tensors="pt")
            
            cur_input_ids = result.pop('input_ids')[0]
            cur_attention_mask = result.pop('attention_mask')[0]
            if 'pixel_values' in result: # 10764, 1176
                pixel_values.append(result["pixel_values"])
            if 'image_grid_thw' in result:
                image_grid_thw.append(result["image_grid_thw"])
            

            input_ids.append(cur_input_ids)
            attention_mask.append(cur_attention_mask)
            if role in ["system", "user"]:
                labels.append(torch.full_like(cur_input_ids, -100))
            else:
                labels.append(cur_input_ids)

        input_ids = torch.cat(input_ids, dim=0)
        labels = torch.cat(labels, dim=0)
        attention_mask = torch.cat(attention_mask, dim=0)  

        self.input_ids = torch.cat([self.input_ids, input_ids], dim=0)
        self.labels = torch.cat([self.labels, labels], dim=0)
        self.attention_mask = torch.cat([self.attention_mask, attention_mask], dim=0)

        pixel_values = torch.cat(pixel_values, dim=0) if len(pixel_values) > 0 else None
        image_grid_thw = torch.cat(image_grid_thw, dim=0) if len(image_grid_thw) > 0 else None

        if self.pixel_values is None:
            self.pixel_values = pixel_values
        else:
            if pixel_values is not None:
                self.pixel_values = torch.cat([self.pixel_values, pixel_values], dim=0)
        
        if self.image_grid_thw is None:
            self.image_grid_thw = image_grid_thw
        else:
            if image_grid_thw is not None:
                self.image_grid_thw = torch.cat([self.image_grid_thw, image_grid_thw], dim=0)
            

    def get_train_dict(self):
        # --- Trim images to match what survives truncation -----------------------
        pixel_values = self.pixel_values
        image_grid_thw = self.image_grid_thw
        max_len = self.config.data.max_prompt_length
        work_ids = self.input_ids.clone()
        work_labels = self.labels.clone()
        work_attn = self.attention_mask.clone()
        n_images_removed = 0

        if pixel_values is not None and image_grid_thw is not None and work_ids.size(0) > max_len:
            truncated_ids = work_ids[:max_len]
            vision_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
            image_pad_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
            vision_end_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
            pad_token_id = self.tokenizer.pad_token_id
            ids_list = truncated_ids.tolist()

            n_surviving_images = 0
            for i, tid in enumerate(ids_list):
                if tid == vision_start_id and i + 1 < len(ids_list) and ids_list[i + 1] == image_pad_id:
                    n_surviving_images += 1

            n_total_images = image_grid_thw.size(0)
            if n_surviving_images > 0 and n_surviving_images <= n_total_images:
                merge_size = self.processor.image_processor.merge_size
                actual_pad_tokens = sum(1 for tid in ids_list if tid == image_pad_id)
                expected_pad_tokens = 0
                for i in range(n_surviving_images):
                    t, h, w = self.image_grid_thw[i].tolist()
                    expected_pad_tokens += (t * h * w) // (merge_size ** 2)
                while n_surviving_images > 0 and actual_pad_tokens < expected_pad_tokens:
                    n_surviving_images -= 1
                    t, h, w = self.image_grid_thw[n_surviving_images].tolist()
                    expected_pad_tokens -= (t * h * w) // (merge_size ** 2)

            if n_surviving_images < n_total_images:
                n_images_removed = n_total_images - n_surviving_images
                image_grid_thw = self.image_grid_thw[:n_surviving_images]
                n_pixels_keep = 0
                for i in range(n_surviving_images):
                    t, h, w = image_grid_thw[i].tolist()
                    n_pixels_keep += t * h * w
                pixel_values = pixel_values[:n_pixels_keep] if n_pixels_keep > 0 else None

                # Scrub orphan image tokens from work_ids
                img_count = 0
                full_ids = work_ids.tolist()
                for j, tid in enumerate(full_ids):
                    if tid == vision_start_id:
                        if j + 1 < len(full_ids) and full_ids[j + 1] == image_pad_id:
                            img_count += 1
                    if img_count > n_surviving_images:
                        if tid in (vision_start_id, image_pad_id, vision_end_id):
                            full_ids[j] = pad_token_id
                work_ids = torch.tensor(full_ids, dtype=work_ids.dtype, device=work_ids.device)
                scrub_mask = (work_ids == pad_token_id) & (self.input_ids != pad_token_id)
                work_attn[scrub_mask] = 0
                work_labels[scrub_mask] = -100

        position_ids = get_rope_index(
                self.processor,
                input_ids=work_ids,
                image_grid_thw=image_grid_thw if n_images_removed > 0 else self.image_grid_thw,
                attention_mask=work_attn,
            )
        
        input_ids, attention_mask, position_ids, labels = VF.postprocess_data(
                input_ids=work_ids,
                attention_mask=work_attn,
                position_ids=position_ids,
                max_length=max_len,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation='right',
                labels=work_labels
            )
        data = {
            'input_ids': input_ids,
            'labels': labels,
            'position_ids': position_ids,
            'attention_mask': attention_mask,
        }
        if pixel_values is not None and image_grid_thw is not None and image_grid_thw.size(0) > 0:
            data['multi_modal_inputs'] = {
                'pixel_values': pixel_values,
                'image_grid_thw': image_grid_thw,
            }
        return data
    
    def reset(self, task_config):

        self.instruction = _instruction_with_cluster_priors(task_config)
        self.task_config = task_config
        self.step_counter = 0
        self.is_done = False

        self.reset_train_tensors()

        trial_time = 0
        while trial_time < 8:
            try:
                obs = self.env.reset(task_config)
                break
            except Exception as e:
                print(f"Env reset exception: {e}")
                print('Env reset error: ', traceback.format_exc())
                trial_time += 1
        
        if trial_time >= 8:
            self.is_init = True
            self.is_done = True
            print('Env reset failed after 8 trials: ', task_config)
            return {
                "env_idx": self.worker_idx,
                "obs_messages": None,
                "is_done": self.is_done,
                'format_reward': 0.0
            }

        # self.agent.reset()
        self.is_init = True


        init_image = obs["screenshot"]
        # init_image = Image.open(BytesIO(init_image))

        image_base64 = base64.b64encode(BytesIO(init_image).getvalue()).decode("utf-8")

        init_messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "Your are a helpful assistant."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self.system_prompt.format(instruction=self.instruction)
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"data:image/jpeg;base64,{image_base64}",
                        "min_pixels": self.min_pixels,
                        "max_pixels": self.max_pixels,
                    }
                ]
            }
        ]

        self.history_images = [init_image]
        self.history_messages = init_messages

        self.process_message(init_messages)

        # since the prediction time can be very long for multienv, we pause the env to avoid automatically locking screen. 
        self.env.pause()
        return {
            'env_idx': self.worker_idx,
            'obs_messages': self.history_messages,
            'is_done': self.is_done,
            'format_reward': 0.0
        }
    
    def step(self, prediction):

        self.is_init = False

        # 1. parse to structure output, might exception
        # 2. parse to pyautogui code, might excpetion
        # 3. env.step, might timeout
        # 4. check the output format(if format is wrong/timeout/exception, return DONE)

        origin_resized_height = obs_image_height = 1080
        origin_resized_width = obs_image_width = 1920

        try:
            parsed_responses = parse_action_to_structure_output(
                prediction,
                self.action_parse_res_factor,
                origin_resized_height,
                origin_resized_width,
                self.model_type,
                self.max_pixels,
                self.min_pixels
            )

            actions = []
            for parsed_response in parsed_responses:
                if "action_type" in parsed_response:
                    if parsed_response["action_type"] == FINISH_WORD:
                        actions = ['DONE']
                        break
                    
                    elif parsed_response["action_type"] == WAIT_WORD:
                        actions = ['WAIT']
                        break
                    
                    elif parsed_response["action_type"] == ENV_FAIL_WORD:
                        actions = ['FAIL']
                        break

                    elif parsed_response["action_type"] == CALL_USER:
                        actions = ['FAIL']
                        break

                pyautogui_code = parsing_response_to_pyautogui_code(
                    parsed_response,
                    obs_image_height,
                    obs_image_width,
                    False # input_swap = False, don't use pyperclip
                )
                actions.append(pyautogui_code)
            
            format_reward = 0.0
        except:
            print('Parse action error: ', prediction)
            print('Error traceback: ', traceback.format_exc())
            format_reward = -1.0
            actions = ['DONE'] # error output format, stop the trajectory immediately

        action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

        self.env.unpause()
        for action in actions:
            obs, reward, step_done, info = self.env.step(action, pause=0.5)
        
            if step_done:
                self.is_done = True
            
            self.step_counter += 1
            if self.step_counter == self.max_steps:
                self.is_done = True

            step_data = {
                "step_num": self.step_counter,
                "action": action,
                "reward": reward,
                "done": step_done,
                "info": info,
                "action_timestamp": action_timestamp,
            }
        
        self.env.pause()

        self.history_messages.append({
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": add_box_token(prediction)
            }]
        })

        if not self.is_done:
            if obs['screenshot'] is None:
                self.is_done = True
                # failed to get screenshot
                self.process_message(self.history_messages[-1:]) # gpt answer only
                return {
                    'env_idx': self.worker_idx,
                    'obs_messages': None,
                    'is_done': self.is_done,
                    'format_reward': format_reward
                }
            self.history_images.append(obs['screenshot'])

            image_base64 = base64.b64encode(BytesIO(obs['screenshot']).getvalue()).decode('utf-8')

            self.history_messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"data:image/jpeg;base64,{image_base64}",
                        "min_pixels": self.min_pixels,
                        "max_pixels": self.max_pixels,
                    }
                ]
            })
            self.process_message(self.history_messages[-2:]) # gpt answer + next image
            return {
                'env_idx': self.worker_idx,
                'obs_messages': self.history_messages,
                'is_done': self.is_done,
                'format_reward': format_reward
            }
        else:
            self.process_message(self.history_messages[-1:]) # gpt answer only
            return {
                'env_idx': self.worker_idx,
                'obs_messages': None,
                'is_done': self.is_done,
                'format_reward': format_reward
            }
            
    
    def evaluate(self):
        try:
            self.env.unpause()
            # we dont care the env after evaluation, since the reset will destroy it and create new env.
            result = self.env.evaluate()
            self._last_eval_result = result
            return result
        except Exception as e:
            print(f"Evaluation error: {e}")
            self._last_eval_result = 0.0
            return 0.0


    def get_history_messages(self):
        return self.history_messages

    def get_history_images(self):
        return self.history_images

    def get_compact_trajectory(self, limit_images: int = 8) -> dict:
        """Return a deduplicated episode dict: each screenshot stored once.

        Each step stores the screenshot the model SAW and the action it took.
        Screenshots are raw JPEG bytes re-encoded as base64 at quality=85.
        """
        import base64
        import io as _io
        from PIL import Image as _Image

        actions = [
            (m["content"][0]["text"] if isinstance(m["content"], list) else m["content"])
            for m in self.history_messages if m.get("role") == "assistant"
        ]
        images = self.history_images  # List of raw JPEG bytes (one per step)

        steps = []
        for img_bytes, action in zip(images, actions):
            # Re-encode as JPEG at quality=85 to save space
            pil = _Image.open(_io.BytesIO(img_bytes))
            buf = _io.BytesIO()
            pil.save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            steps.append({"screenshot_b64": b64, "action": action})

        return {
            "task_id": (self.task_config.get("task_id") or self.task_config.get("id", "")) if self.task_config else "",
            "instruction": self.instruction or "",
            "eval_result": getattr(self, "_last_eval_result", 0.0),
            "limit_images": limit_images,
            "steps": steps,
        }

    def is_done(self):
        return self.is_done

    def is_init(self):
        return self.is_init


# --- Remote env worker: same interface as EnvWorker but talks to HTTP server (Mac/AWS) ---
import requests
from .remote_env_protocol import wire_to_messages


def _instruction_with_cluster_priors(task_config: dict) -> str:
    """Inject compact task-family priors on the cluster side before generation."""
    text = (task_config.get("instruction") or "").strip()
    domain = str(task_config.get("domain") or "").strip().lower()
    lower = text.lower()
    hints = []

    if domain == "gimp" or ("photo" in lower and any(k in lower for k in ("brightness", "vibrancy", "color"))):
        hints.append("Use GIMP/image editor tools, not Ubuntu System Settings.")
        hints.append("If GIMP shows a blank canvas, use File -> Open to load the image before using Colors adjustments.")
        hints.append("For color tasks, prefer Colors menu actions (e.g., Color Balance/Saturation) over random toolbar/sidebar clicks.")
    if "bing" in lower and "search" in lower:
        hints.append("Default search engine changes are browser settings (Chrome/Firefox), not Ubuntu system settings.")
    if any(k in lower for k in ("search", "discussions", "browser", "website")):
        hints.append("Browser search flow: click address bar/search field, type query, press Enter; avoid repeated new-tab/menu clicks if current tab works.")
    if "shortcut" in lower and any(k in lower for k in ("site", "website", "page")):
        hints.append("Create a browser/webpage shortcut on the Desktop, not a generic file/folder in terminal.")
        hints.append("Do not default to terminal/file-manager workflows unless the task explicitly asks for terminal commands.")
    if "trash" in lower and any(k in lower for k in ("recover", "deleted")):
        hints.append("Use Files/Trash and restore the target file; avoid terminal unless the task asks for it.")
        hints.append("After opening Trash, look for the target file and use Restore/Restore From Trash; avoid repeated clicks on the same list item.")
    if domain == "vlc" or "vlc" in lower or "music video" in lower:
        hints.append("Use VLC/media-player controls and the Desktop file, not system settings or search loops.")
        hints.append("Typical VLC flow: open VLC, then Media -> Open File (or open the Desktop file directly in VLC).")

    if hints:
        return text + "\n\nTask-family guidance:\n- " + "\n- ".join(hints)
    return text


@ray.remote(num_cpus=1)
class RemoteEnvWorker:
    """Env worker that forwards reset/step/evaluate to a remote HTTP server (one env on Mac/AWS)."""
    system_prompt = uitars_system_prompt

    def __init__(self, worker_idx, max_steps, config, remote_server_url: str, slot_id: int = None):
        import os
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        self.worker_idx = worker_idx
        self.slot_id = slot_id if slot_id is not None else worker_idx  # local slot on the remote server
        self.max_steps = max_steps
        self.config = config
        self.remote_server_url = remote_server_url.rstrip("/")
        self.step_counter = 0
        self._is_done = False
        self._is_init = False
        self.instruction = None
        self.task_config = None
        self.history_messages = []
        self.history_images = []

        self.tokenizer = get_tokenizer(
            config.worker.actor.model.model_path,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )
        self.processor = get_processor(
            config.worker.actor.model.model_path,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )
        self.action_parse_res_factor = 1000
        self.model_type = "qwen25vl"
        self.max_pixels = config.data.max_pixels
        self.min_pixels = config.data.min_pixels
        self.reset_train_tensors()

    def reset_train_tensors(self):
        self.input_ids = torch.zeros((0,), dtype=torch.int64)
        self.labels = torch.full((0,), -100, dtype=torch.int64)
        self.attention_mask = torch.zeros((0,), dtype=torch.int64)
        self.pixel_values = None
        self.image_grid_thw = None

    def load_content(self, content):
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join([self.load_content(c) for c in content])
        if isinstance(content, dict):
            if "text" in content:
                return content["text"]
            if "image" in content:
                return "<|vision_start|><|image_pad|><|vision_end|>"
        raise ValueError(f"Unknown content type: {content}")

    def process_message(self, message):
        tokenizer = self.tokenizer
        processor = self.processor
        image_inputs, video_inputs, video_kwargs = process_vision_info(message, return_video_kwargs=True)
        input_ids, labels, attention_mask = [], [], []
        image_count = 0
        pixel_values, image_grid_thw = [], []
        for msg in message:
            role = msg["role"]
            content = self.load_content(msg["content"])
            prompt = f"<|im_start|>{role}\n" + content + "<|im_end|>\n"
            cur_image_num = prompt.count("<|image_pad|>")
            if cur_image_num > 0:
                result = processor(
                    image_inputs[image_count : image_count + cur_image_num],
                    [prompt],
                    add_special_tokens=False,
                    return_tensors="pt",
                )
                image_count += cur_image_num
            else:
                result = processor(None, [prompt], add_special_tokens=False, return_tensors="pt")
            cur_input_ids = result.pop("input_ids")[0]
            cur_attention_mask = result.pop("attention_mask")[0]
            if "pixel_values" in result:
                pixel_values.append(result["pixel_values"])
            if "image_grid_thw" in result:
                image_grid_thw.append(result["image_grid_thw"])
            input_ids.append(cur_input_ids)
            attention_mask.append(cur_attention_mask)
            labels.append(torch.full_like(cur_input_ids, -100) if role in ("system", "user") else cur_input_ids)
        input_ids = torch.cat(input_ids, dim=0)
        labels = torch.cat(labels, dim=0)
        attention_mask = torch.cat(attention_mask, dim=0)
        self.input_ids = torch.cat([self.input_ids, input_ids], dim=0)
        self.labels = torch.cat([self.labels, labels], dim=0)
        self.attention_mask = torch.cat([self.attention_mask, attention_mask], dim=0)
        pv = torch.cat(pixel_values, dim=0) if pixel_values else None
        igt = torch.cat(image_grid_thw, dim=0) if image_grid_thw else None
        if self.pixel_values is None:
            self.pixel_values = pv
        elif pv is not None:
            self.pixel_values = torch.cat([self.pixel_values, pv], dim=0)
        if self.image_grid_thw is None:
            self.image_grid_thw = igt
        elif igt is not None:
            self.image_grid_thw = torch.cat([self.image_grid_thw, igt], dim=0)

    def get_train_dict(self):
        if self.processor is None:
            return {
                "input_ids": self.input_ids,
                "labels": self.labels,
                "position_ids": torch.zeros((3, self.input_ids.size(0)), dtype=torch.int64) if self.input_ids.numel() else torch.zeros((3, 0), dtype=torch.int64),
                "attention_mask": self.attention_mask,
            }

        # --- Trim images to match what survives truncation -----------------------
        pixel_values = self.pixel_values
        image_grid_thw = self.image_grid_thw
        max_len = self.config.data.max_prompt_length
        # Work on a copy so self.input_ids/labels/attention_mask aren't mutated
        work_ids = self.input_ids.clone()
        work_labels = self.labels.clone()
        work_attn = self.attention_mask.clone()
        n_images_removed = 0

        if pixel_values is not None and image_grid_thw is not None and work_ids.size(0) > max_len:
            truncated_ids = work_ids[:max_len]
            vision_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
            image_pad_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
            vision_end_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
            pad_token_id = self.tokenizer.pad_token_id
            ids_list = truncated_ids.tolist()

            # Count images whose <|vision_start|> appears in truncated window
            n_surviving_images = 0
            for i, tid in enumerate(ids_list):
                if tid == vision_start_id and i + 1 < len(ids_list) and ids_list[i + 1] == image_pad_id:
                    n_surviving_images += 1

            n_total_images = image_grid_thw.size(0)
            if n_surviving_images > 0 and n_surviving_images <= n_total_images:
                merge_size = self.processor.image_processor.merge_size
                actual_pad_tokens = sum(1 for tid in ids_list if tid == image_pad_id)
                expected_pad_tokens = 0
                for i in range(n_surviving_images):
                    t, h, w = self.image_grid_thw[i].tolist()
                    expected_pad_tokens += (t * h * w) // (merge_size ** 2)
                while n_surviving_images > 0 and actual_pad_tokens < expected_pad_tokens:
                    n_surviving_images -= 1
                    t, h, w = self.image_grid_thw[n_surviving_images].tolist()
                    expected_pad_tokens -= (t * h * w) // (merge_size ** 2)

            if n_surviving_images < n_total_images:
                n_images_removed = n_total_images - n_surviving_images
                image_grid_thw = self.image_grid_thw[:n_surviving_images]
                n_pixels_keep = 0
                for i in range(n_surviving_images):
                    t, h, w = image_grid_thw[i].tolist()
                    n_pixels_keep += t * h * w
                pixel_values = pixel_values[:n_pixels_keep] if n_pixels_keep > 0 else None

                # Scrub orphan image tokens from work_ids so the model doesn't see
                # image_pad tokens without matching pixel features.
                # Walk work_ids, count images by vision_start; once past n_surviving_images,
                # replace vision_start/image_pad/vision_end with pad_token_id.
                img_count = 0
                full_ids = work_ids.tolist()
                for j, tid in enumerate(full_ids):
                    if tid == vision_start_id:
                        # Check if this is an image start (followed by image_pad)
                        if j + 1 < len(full_ids) and full_ids[j + 1] == image_pad_id:
                            img_count += 1
                    if img_count > n_surviving_images:
                        if tid in (vision_start_id, image_pad_id, vision_end_id):
                            full_ids[j] = pad_token_id
                work_ids = torch.tensor(full_ids, dtype=work_ids.dtype, device=work_ids.device)
                # Also zero out attention_mask and labels for the scrubbed positions
                scrub_mask = (work_ids == pad_token_id) & (self.input_ids != pad_token_id)
                work_attn[scrub_mask] = 0
                work_labels[scrub_mask] = -100

        # Compute position_ids with original full-sequence data for correct rope
        position_ids = get_rope_index(
            self.processor,
            input_ids=work_ids,
            image_grid_thw=image_grid_thw if n_images_removed > 0 else self.image_grid_thw,
            attention_mask=work_attn,
        )
        input_ids, attention_mask, position_ids, labels = VF.postprocess_data(
            input_ids=work_ids,
            attention_mask=work_attn,
            position_ids=position_ids,
            max_length=max_len,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation="right",
            labels=work_labels,
        )
        data = {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }
        if pixel_values is not None and image_grid_thw is not None and image_grid_thw.size(0) > 0:
            data["multi_modal_inputs"] = {
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
            }
        return data

    # Longer timeouts for remote env (Mac/CPU or KVM); evaluate can be slow
    REMOTE_RESET_TIMEOUT = 1200   # 20 min
    REMOTE_STEP_TIMEOUT = 300     # 5 min
    REMOTE_EVALUATE_TIMEOUT = 300  # 5 min
    REMOTE_EVALUATE_RETRIES = 4   # 5 attempts total; 503/env-not-ready often transient
    REMOTE_RESET_RETRIES = 2      # 3 attempts total for reset (503 often transient)

    def _post(self, path: str, json_body: dict, timeout=None):
        url = f"{self.remote_server_url}{path}"
        r = requests.post(url, json=json_body, timeout=timeout)
        r.raise_for_status()
        return r.json()

    def _task_config_with_noise(self, task_config: dict) -> dict:
        cfg = dict(task_config)
        algo = getattr(self.config, "algorithm", None)
        if algo is None or not getattr(algo, "enable_noise", False):
            return cfg

        cfg["enable_noise"] = True
        cfg["noise_mode"] = getattr(algo, "noise_mode", "task_noise_meta")
        cfg["noise_probability"] = float(getattr(algo, "noise_probability", 0.0))
        cfg["noise_tier"] = int(getattr(algo, "noise_initial_tier", 1))
        cfg["noise_use_heldout"] = bool(getattr(algo, "noise_use_heldout", False))
        return cfg

    def reset(self, task_config):
        import time
        task_config = self._task_config_with_noise(task_config)
        self.instruction = _instruction_with_cluster_priors(task_config)
        self.task_config = task_config
        self.step_counter = 0
        self._is_done = False
        self._is_init = True
        self.reset_train_tensors()
        self.history_messages = []
        self.history_images = []

        # Phase 1: HTTP POST /env/reset with retries for network/server failures.
        last_err = None
        resp = None
        for attempt in range(self.REMOTE_RESET_RETRIES + 1):
            try:
                resp = self._post("/env/reset", {"task_config": task_config, "slot_id": self.slot_id}, timeout=self.REMOTE_RESET_TIMEOUT)
                break
            except Exception as e:
                last_err = e
                print(f"RemoteEnvWorker[{self.worker_idx}] reset HTTP attempt {attempt+1} failed: {type(e).__name__}: {e}")
                if attempt < self.REMOTE_RESET_RETRIES:
                    time.sleep(5 * (attempt + 1))
        else:
            print(f"RemoteEnvWorker[{self.worker_idx}] reset HTTP failed (all {self.REMOTE_RESET_RETRIES + 1} attempts): {last_err}.")
            return {"env_idx": self.worker_idx, "obs_messages": None, "is_done": True, "format_reward": 0.0}

        # Phase 2: Process the screenshot locally.  Server-side validation should
        # have caught most truncated images, but network transfer can still corrupt.
        # Retry process_message WITHOUT re-resetting the env (env state is fine).
        obs_wire = resp.get("obs_messages")
        if obs_wire:
            pm_err = None
            for pm_attempt in range(self.REMOTE_RESET_RETRIES + 1):
                try:
                    # Re-clear train tensors before each attempt so a partial
                    # first attempt doesn't leave stale data that gets doubled.
                    self.reset_train_tensors()
                    self.history_messages = wire_to_messages(obs_wire)
                    self.history_images = self._extract_images_from_messages(self.history_messages)
                    self.process_message(self.history_messages)
                    pm_err = None
                    break
                except Exception as e:
                    pm_err = e
                    print(f"RemoteEnvWorker[{self.worker_idx}] process_message attempt {pm_attempt+1} failed: {type(e).__name__}: {e}")
                    if pm_attempt < self.REMOTE_RESET_RETRIES:
                        time.sleep(2 * (pm_attempt + 1))
            if pm_err is not None:
                print(f"RemoteEnvWorker[{self.worker_idx}] process_message failed after all retries; marking done.")
                self.history_messages = []
                return {"env_idx": self.worker_idx, "obs_messages": None, "is_done": True, "format_reward": 0.0}

        self._is_done = resp.get("is_done", True)
        return {
            "env_idx": self.worker_idx,
            "obs_messages": self.history_messages if obs_wire else None,
            "is_done": self._is_done,
            "format_reward": resp.get("format_reward", 0.0),
        }

    def step(self, prediction):
        self._is_init = False
        algo = getattr(self.config, "algorithm", None)
        noise_probability = float(getattr(algo, "noise_probability", 0.0)) if algo is not None else 0.0
        try:
            resp = self._post(
                "/env/step",
                {"prediction": prediction, "slot_id": self.slot_id, "noise_probability": noise_probability},
                timeout=self.REMOTE_STEP_TIMEOUT,
            )
        except Exception as e:
            print(f"RemoteEnvWorker step HTTP error: {e}")
            return {"env_idx": self.worker_idx, "obs_messages": None, "is_done": True, "format_reward": -1.0}

        self._is_done = resp.get("is_done", True)
        obs_wire = resp.get("obs_messages")
        if obs_wire:
            self.history_messages = wire_to_messages(obs_wire)
            # Extract new screenshot(s) from the latest user message(s).
            # When is_done=True (e.g. "finished()" action), the server returns
            # the full history WITHOUT a new screenshot — skip extraction to
            # avoid duplicating the previous step's screenshot.
            if not self._is_done:
                new_imgs = self._extract_images_from_messages(self.history_messages[-2:])
                self.history_images.extend(new_imgs)
            try:
                self.process_message(self.history_messages[-2:])
            except Exception as e:
                print(f"RemoteEnvWorker[{self.worker_idx}] step: process_message failed ({type(e).__name__}: {e}); marking done")
                return {"env_idx": self.worker_idx, "obs_messages": None, "is_done": True, "format_reward": -1.0}
        return {
            "env_idx": self.worker_idx,
            "obs_messages": self.history_messages if obs_wire else None,
            "is_done": self._is_done,
            "format_reward": resp.get("format_reward", 0.0),
        }

    def evaluate(self):
        import time
        last_err = None
        for attempt in range(self.REMOTE_EVALUATE_RETRIES + 1):
            try:
                result = self._post("/env/evaluate", {"slot_id": self.slot_id}, timeout=self.REMOTE_EVALUATE_TIMEOUT)
                score = result["score"] if isinstance(result, dict) else result
                score_float = float(score)
                print(f"RemoteEnvWorker[{self.worker_idx}] evaluate: score={score_float}, instruction={self.instruction}")
                self._last_eval_result = score_float
                return score_float
            except Exception as e:
                last_err = e
                if attempt == 0:
                    print(f"RemoteEnvWorker evaluate failed: {e}. Retrying up to {self.REMOTE_EVALUATE_RETRIES} times with backoff...")
                if attempt < self.REMOTE_EVALUATE_RETRIES:
                    wait = 5 * (attempt + 1)  # 5s, 10s, 15s, 20s backoff
                    time.sleep(wait)
        print(f"RemoteEnvWorker evaluate HTTP error (all {self.REMOTE_EVALUATE_RETRIES + 1} attempts failed): {last_err}. Returning 0.0.")
        self._last_eval_result = 0.0
        return 0.0

    @staticmethod
    def _extract_images_from_messages(messages) -> list:
        """Extract raw JPEG bytes from message image content (base64 data URLs)."""
        import base64 as _b64
        images = []
        for m in messages:
            if m.get("role") != "user":
                continue
            for c in (m.get("content", []) if isinstance(m.get("content"), list) else []):
                if isinstance(c, dict) and c.get("type") == "image" and "image" in c:
                    img_str = c["image"]
                    if "," in img_str:
                        img_str = img_str.split(",", 1)[1]
                    images.append(_b64.b64decode(img_str))
        return images

    def get_history_messages(self):
        return self.history_messages

    def get_history_images(self):
        return self.history_images

    def get_compact_trajectory(self, limit_images: int = 8) -> dict:
        """Return a deduplicated episode dict: each screenshot stored once.

        Mirrors EnvWorker.get_compact_trajectory(). Uses history_images (raw JPEG
        bytes) and extracts actions from history_messages assistant turns.
        """
        import base64
        import io as _io
        from PIL import Image as _Image

        actions = [
            (m["content"][0]["text"] if isinstance(m["content"], list) else m["content"])
            for m in self.history_messages if m.get("role") == "assistant"
        ]
        images = self.history_images  # List of raw JPEG bytes (one per step)

        steps = []
        for img_bytes, action in zip(images, actions):
            pil = _Image.open(_io.BytesIO(img_bytes))
            buf = _io.BytesIO()
            pil.save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            steps.append({"screenshot_b64": b64, "action": action})

        return {
            "task_id": (self.task_config.get("task_id") or self.task_config.get("id", "")) if self.task_config else "",
            "instruction": self.instruction or "",
            "eval_result": getattr(self, "_last_eval_result", 0.0),
            "limit_images": limit_images,
            "steps": steps,
        }

    def is_done(self):
        return self._is_done

    def is_init(self):
        return self._is_init
