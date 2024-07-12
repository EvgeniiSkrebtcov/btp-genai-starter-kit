import logging
import os
from pydantic import BaseModel
import requests

log = logging.getLogger(__name__)


class Image(BaseModel):
    src: str
    mime_type: str
    data: str


class ImageProcessor:
    def __init__(self, image: Image):
        if not image.src or not image.mime_type or not image.data:
            raise ValueError("Image attributes must be set")
        self.image = image

    def execute(self, prompt: str, auth_token: str) -> str:
        api_base = os.environ.get("AICORE_BASE_URL")
        deployment_id = os.environ.get("AICORE_DEPLOYMENT")
        resource_group = os.environ.get("AICORE_RESOURCE_GROUP") or "default"

        api_url = f"{api_base}/v2/inference/deployments/{deployment_id}/invoke"
        # Create the headers for the request
        headers = {
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json",
            "AI-Resource-Group": resource_group,
        }

        data = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4000,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": self.image.mime_type,
                                "data": self.image.data,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }

        response = requests.post(url=api_url, headers=headers, json=data, timeout=20)

        response_as_json = response.json()

        log.info("Image from %s was successfully processed", self.image.src)

        return response_as_json["content"][0]["text"]
