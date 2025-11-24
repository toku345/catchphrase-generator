import json
import os
from typing import Optional

import boto3
from botocore.exceptions import ClientError


class BedrockClientError(Exception):
    """Raised when Amazon Bedrock catchphrase generation fails."""


class BedrockClient:
    """Client for Amazon Bedrock Nova micro model"""

    def __init__(
        self,
        region: Optional[str] = None,
        model_id: Optional[str] = None
    ):
        self.region = region or os.environ.get("AWS_REGION_BEDROCK", "us-east-1")
        self.model_id = model_id or os.environ.get(
            "BEDROCK_MODEL_ID",
            "us.amazon.nova-micro-v1:0"
        )
        self.client = boto3.client("bedrock-runtime", region_name=self.region)

    def generate_catchphrase(
        self,
        starting_word: Optional[str] = None,
        background: Optional[str] = None,
        style: Optional[str] = None
    ) -> str:
        """
        Generate catchphrase using Amazon Bedrock Nova micro

        Args:
            starting_word: The word to start the catchphrase with
            background: Background context for the catchphrase
            style: Style or tone for the catchphrase

        Returns:
            Generated catchphrase string

        Raises:
            BedrockClientError: If Bedrock API call fails
        """
        prompt = self._build_prompt(starting_word, background, style)

        request_body = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": prompt}]
                }
            ],
            "inferenceConfig": {
                "max_new_tokens": 200,
                "temperature": 0.7,
                "top_p": 0.9
            }
        }

        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body)
            )

            response_body = json.loads(response["body"].read())
            catchphrase = response_body["output"]["message"]["content"][0]["text"]

            return catchphrase.strip()

        except ClientError as err:
            raise BedrockClientError("Failed to generate catchphrase") from err

    def _build_prompt(
        self,
        starting_word: Optional[str],
        background: Optional[str],
        style: Optional[str]
    ) -> str:
        """Build prompt for catchphrase generation"""
        prompt_parts = [
            "あなたはクリエイティブなキャッチフレーズを生成するアシスタントです。",
            "以下の条件に基づいて、魅力的なキャッチフレーズを1つだけ生成してください。",
            ""
        ]

        if background:
            prompt_parts.append(f"背景・コンテキスト: {background}")

        if starting_word:
            prompt_parts.append(f"冒頭の単語: 「{starting_word}」で始めてください")

        if style:
            prompt_parts.append(f"スタイル・トーン: {style}")

        prompt_parts.extend([
            "",
            "キャッチフレーズのみを出力してください。説明や追加のテキストは不要です。"
        ])

        return "\n".join(prompt_parts)
