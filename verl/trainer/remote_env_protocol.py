"""
HTTP protocol for remote OSWorld env (cluster <-> Mac/AWS server).

POST /env/reset   Body: {"task_config": {...}}  -> {env_idx, obs_messages, is_done, format_reward}
POST /env/step    Body: {"prediction": "..."}   -> same shape
POST /env/evaluate Body: {}                     -> float (JSON number)
POST /env/history_messages Body: {}              -> {history_messages: [...]}

Images on wire: "b64" key (raw base64); client uses "data:image/jpeg;base64," + b64.
"""

from typing import Dict, List

IMAGE_DATA_URL_PREFIX = "data:image/jpeg;base64,"


def message_content_to_wire(content: List[Dict]) -> List[Dict]:
    """Convert message content for JSON (extract b64 from data URL)."""
    out = []
    for c in content:
        if c.get("type") == "image" and "image" in c:
            img = c["image"]
            if img.startswith(IMAGE_DATA_URL_PREFIX):
                b64 = img[len(IMAGE_DATA_URL_PREFIX):]
            else:
                b64 = img
            out.append({
                "type": "image",
                "b64": b64,
                "min_pixels": c.get("min_pixels", 3136),
                "max_pixels": c.get("max_pixels", 2116800),
            })
        else:
            out.append(c)
    return out


def wire_content_to_message(content: List[Dict]) -> List[Dict]:
    """Convert wire content back to message content (add data URL prefix)."""
    out = []
    for c in content:
        if c.get("type") == "image" and "b64" in c:
            out.append({
                "type": "image",
                "image": IMAGE_DATA_URL_PREFIX + c["b64"],
                "min_pixels": c.get("min_pixels", 3136),
                "max_pixels": c.get("max_pixels", 2116800),
            })
        else:
            out.append(c)
    return out


def messages_to_wire(messages: List[Dict]) -> List[Dict]:
    """Convert obs_messages for JSON response."""
    return [
        {"role": m["role"], "content": message_content_to_wire(m["content"])}
        for m in messages
    ]


def wire_to_messages(wire: List[Dict]) -> List[Dict]:
    """Convert wire list back to obs_messages for processor."""
    return [
        {"role": m["role"], "content": wire_content_to_message(m["content"])}
        for m in wire
    ]
