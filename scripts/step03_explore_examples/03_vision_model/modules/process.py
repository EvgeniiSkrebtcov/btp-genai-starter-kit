import logging
import sys
import os
from typing import List
import requests
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
from .config import AZURE_LLM, AWS_LLM

log = logging.getLogger(__name__)


class Image(BaseModel):
    src: str
    mime_type: str
    data: str


class ImageProcessorBase:
    def is_valid_image(self, image: Image):
        if not image:
            raise ValueError("Image must be set")
        if not image.src:
            raise ValueError("Image source must be set")
        if not image.mime_type:
            raise ValueError("Image mime type must be set")
        if not image.data:
            raise ValueError("Image data must be set")

    def get_prompt(self) -> str:
        raise NotImplementedError("Method get_prompt must be implemented")

    def execute_via_sdk(self, messages: List[dict]) -> str:
        try:
            proxy_client = get_proxy_client("gen-ai-hub")

            llm = ChatOpenAI(
                proxy_model_name=AZURE_LLM,
                proxy_client=proxy_client,
                temperature=0,
            )
            response = llm.invoke(messages)
            return response.content
        except Exception as e:
            log.error(f"Error executing via SDK: {str(e)}")
            sys.exit()

    def execute_via_http(self, messages: List[dict], auth_token: str) -> str:
        try:
            api_base = os.environ.get("AICORE_BASE_URL")
            deployment_id = AWS_LLM
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
                "messages": messages,
            }

            log.info("Executing image processing...")

            response = requests.post(
                url=api_url, headers=headers, json=data, timeout=20
            )

            response_as_json = response.json()

            log.info("Image processed")

            return response_as_json["content"][0]["text"]
        except requests.exceptions.RequestException as e:
            log.error(f"Error processing image via plain http: {str(e)}")
            sys.exit()


class TabularDataImageProcessor(ImageProcessorBase):
    def __init__(self, image):
        super().is_valid_image(image)
        self.image = image

    def get_prompt(self) -> str:
        return """
            Search for information listed as a table or as list.
            Remove any unnecessary information for example trailing characters.
            Read the name, the values and unit of measurement of the values.
            The output should be a table using markdown as format.
            Come up with meaningful column names if the names are not provided on the image.
            Scan the image line by line and think step by step.
            The output should contain ONLY the markdown table WITHOUT any additional explanations.
        """

    def execute(self, auth_token: str) -> str:
        prompt = self.get_prompt()
        messages = [
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
        ]

        return super().execute_via_http(messages, auth_token)


class VisualReasoningProcessor(ImageProcessorBase):
    def __init__(self, image):
        super().is_valid_image(image)
        self.image = image

    def get_prompt(self) -> str:
        return """
            Act as supervisor who is responsible to detect potential risks based on the information provided in an image.
            In case if any risks hav been detected give me a detailed explanation of the risks and the potential impact.
            The output should be a bullet list using markdown format containing the following list items:
            - Risk Detected (Yes, No)
            - Potential Impact
            - Mitigation Strategy
            - Risk Level (Low, Medium, High)
            Reason step by step about the potential outcome.
            If no risks have been detected, return a message that no risks have been found.
            Provite the result WITHOUT any additional explanations.
        """

    def execute(self) -> str:
        prompt = self.get_prompt()

        return super().execute_via_sdk(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{self.image.data}"
                            },
                        },
                    ]
                )
            ]
        )
