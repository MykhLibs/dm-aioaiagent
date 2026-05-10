import base64
import mimetypes
from pathlib import Path
from typing import Optional, Union

from langchain_core.messages import HumanMessage


class InputImage:
    """Cross-provider image input helper.

    Builds a ``HumanMessage`` whose ``.content`` is a list of LangChain v1
    standard content blocks — one optional ``{"type":"text"}`` followed by
    one ``{"type":"image"}``. The same message is accepted by OpenAI,
    Anthropic, and Gemini chat models when used through ``init_chat_model``.
    """

    @classmethod
    def from_file(
        cls,
        path: Union[str, Path],
        *,
        text: Optional[str] = None,
        mime_type: Optional[str] = None,
    ) -> HumanMessage:
        path = Path(path)
        data = path.read_bytes()
        if mime_type is None:
            mime_type, _ = mimetypes.guess_type(str(path))
            if mime_type is None:
                raise ValueError(
                    f"Could not infer mime type for {path!s}. Pass mime_type explicitly."
                )
        return cls.from_bytes(data, mime_type=mime_type, text=text)

    @classmethod
    def from_url(cls, url: str, *, text: Optional[str] = None) -> HumanMessage:
        block = {"type": "image", "url": str(url)}
        return cls._build_message(block, text)

    @classmethod
    def from_base64(
        cls,
        data: str,
        *,
        mime_type: str,
        text: Optional[str] = None,
    ) -> HumanMessage:
        if not mime_type:
            raise ValueError("mime_type is required for from_base64().")
        block = {"type": "image", "base64": str(data), "mime_type": str(mime_type)}
        return cls._build_message(block, text)

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        mime_type: str,
        text: Optional[str] = None,
    ) -> HumanMessage:
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("from_bytes() expects bytes-like object.")
        b64 = base64.b64encode(bytes(data)).decode("ascii")
        return cls.from_base64(b64, mime_type=mime_type, text=text)

    @staticmethod
    def _build_message(image_block: dict, text: Optional[str]) -> HumanMessage:
        content: list[dict] = []
        if text:
            content.append({"type": "text", "text": str(text)})
        content.append(image_block)
        return HumanMessage(content=content)
