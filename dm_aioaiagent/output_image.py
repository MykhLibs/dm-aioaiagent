import base64
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

from langchain_core.messages import BaseMessage


@dataclass
class OutputImage:
    """Provider-agnostic representation of an AI-generated image."""

    bytes: bytes
    mime_type: str

    def save(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.write_bytes(self.bytes)
        return path

    def to_base64(self) -> str:
        return base64.b64encode(self.bytes).decode("ascii")

    def __repr__(self) -> str:
        return f"OutputImage(mime_type={self.mime_type!r}, size={len(self.bytes)} bytes)"

    @classmethod
    def extract_from(cls, message: BaseMessage) -> List["OutputImage"]:
        """Extract every image block from ``message.content_blocks``.

        Inline base64 blocks decode directly. URL blocks are fetched via
        ``urllib`` and the response's ``Content-Type`` is preferred over the
        block's declared ``mime_type``. Blocks that fail to decode or download
        are silently skipped — partial extraction beats raising on one bad block.
        """
        result: List["OutputImage"] = []
        for block in message.content_blocks:
            if not isinstance(block, dict) or block.get("type") != "image":
                continue
            mime_type = block.get("mime_type") or "image/png"
            if block.get("base64"):
                try:
                    data = base64.b64decode(block["base64"])
                except Exception:
                    continue
            elif block.get("url"):
                try:
                    with urllib.request.urlopen(block["url"]) as resp:
                        data = resp.read()
                        ct = resp.headers.get("Content-Type")
                        if ct:
                            mime_type = ct.split(";", 1)[0].strip()
                except Exception:
                    continue
            else:
                continue
            result.append(cls(bytes=data, mime_type=mime_type))
        return result
