class OpenAIImageMessageContent(list):
    def __init__(self, image_url: str, text: str = None):
        content = []
        if isinstance(text, str):
            content.append({
                "type": "text",
                "text": text
            })
        content.append({
            "type": "image_url",
            "image_url": {
                "url": image_url
            }
        })
        super().__init__(content)
