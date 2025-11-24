import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

# Add src directory to Python path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from bedrock_client import BedrockClient, BedrockClientError


class TestBedrockClient:
    """Test cases for BedrockClient"""

    def test_successful_generation(self):
        """Test successful catchphrase generation with valid response"""
        client = BedrockClient()

        mock_response = {
            "body": MagicMock(read=lambda: json.dumps({
                "output": {
                    "message": {
                        "content": [
                            {"text": "  テストキャッチフレーズ  "}
                        ]
                    }
                }
            }).encode())
        }

        with patch.object(client.client, 'invoke_model', return_value=mock_response):
            result = client.generate_catchphrase(
                starting_word="未来",
                background="テクノロジー企業",
                style="前向き"
            )

        assert result == "テストキャッチフレーズ"

    def test_invalid_json_response(self):
        """Test handling of invalid JSON in response"""
        client = BedrockClient()

        mock_response = {
            "body": MagicMock(read=lambda: b"invalid json")
        }

        with patch.object(client.client, 'invoke_model', return_value=mock_response):
            with pytest.raises(BedrockClientError, match="Failed to decode Bedrock response"):
                client.generate_catchphrase(background="テスト")

    def test_missing_output_field(self):
        """Test handling of missing 'output' field"""
        client = BedrockClient()

        mock_response = {
            "body": MagicMock(read=lambda: json.dumps({
                "result": {}
            }).encode())
        }

        with patch.object(client.client, 'invoke_model', return_value=mock_response):
            with pytest.raises(BedrockClientError, match="Missing or invalid 'output'"):
                client.generate_catchphrase(background="テスト")

    def test_invalid_output_type(self):
        """Test handling of invalid 'output' type"""
        client = BedrockClient()

        mock_response = {
            "body": MagicMock(read=lambda: json.dumps({
                "output": "not a dict"
            }).encode())
        }

        with patch.object(client.client, 'invoke_model', return_value=mock_response):
            with pytest.raises(BedrockClientError, match="Missing or invalid 'output'"):
                client.generate_catchphrase(background="テスト")

    def test_missing_message_field(self):
        """Test handling of missing 'message' field"""
        client = BedrockClient()

        mock_response = {
            "body": MagicMock(read=lambda: json.dumps({
                "output": {}
            }).encode())
        }

        with patch.object(client.client, 'invoke_model', return_value=mock_response):
            with pytest.raises(BedrockClientError, match="Missing or invalid 'message'"):
                client.generate_catchphrase(background="テスト")

    def test_empty_content_list(self):
        """Test handling of empty content list"""
        client = BedrockClient()

        mock_response = {
            "body": MagicMock(read=lambda: json.dumps({
                "output": {
                    "message": {
                        "content": []
                    }
                }
            }).encode())
        }

        with patch.object(client.client, 'invoke_model', return_value=mock_response):
            with pytest.raises(BedrockClientError, match="Missing or invalid 'content'"):
                client.generate_catchphrase(background="テスト")

    def test_invalid_content_type(self):
        """Test handling of invalid content type (not a list)"""
        client = BedrockClient()

        mock_response = {
            "body": MagicMock(read=lambda: json.dumps({
                "output": {
                    "message": {
                        "content": "not a list"
                    }
                }
            }).encode())
        }

        with patch.object(client.client, 'invoke_model', return_value=mock_response):
            with pytest.raises(BedrockClientError, match="Missing or invalid 'content'"):
                client.generate_catchphrase(background="テスト")

    def test_missing_text_field(self):
        """Test handling of missing 'text' field"""
        client = BedrockClient()

        mock_response = {
            "body": MagicMock(read=lambda: json.dumps({
                "output": {
                    "message": {
                        "content": [
                            {"data": "something else"}
                        ]
                    }
                }
            }).encode())
        }

        with patch.object(client.client, 'invoke_model', return_value=mock_response):
            with pytest.raises(BedrockClientError, match="Missing or invalid 'text'"):
                client.generate_catchphrase(background="テスト")

    def test_invalid_text_type(self):
        """Test handling of invalid text type (not a string)"""
        client = BedrockClient()

        mock_response = {
            "body": MagicMock(read=lambda: json.dumps({
                "output": {
                    "message": {
                        "content": [
                            {"text": 123}
                        ]
                    }
                }
            }).encode())
        }

        with patch.object(client.client, 'invoke_model', return_value=mock_response):
            with pytest.raises(BedrockClientError, match="Missing or invalid 'text'"):
                client.generate_catchphrase(background="テスト")

    def test_client_error_handling(self):
        """Test handling of ClientError from Bedrock API"""
        client = BedrockClient()

        error_response = {'Error': {'Code': '500', 'Message': 'Internal Server Error'}}
        client_error = ClientError(error_response, 'InvokeModel')

        with patch.object(client.client, 'invoke_model', side_effect=client_error):
            with pytest.raises(BedrockClientError, match="Failed to generate catchphrase"):
                client.generate_catchphrase(background="テスト")

    def test_invalid_response_body_type(self):
        """Test handling of non-dict response body"""
        client = BedrockClient()

        mock_response = {
            "body": MagicMock(read=lambda: json.dumps([]).encode())
        }

        with patch.object(client.client, 'invoke_model', return_value=mock_response):
            with pytest.raises(BedrockClientError, match="Invalid response type"):
                client.generate_catchphrase(background="テスト")

    def test_whitespace_trimming(self):
        """Test that whitespace is properly trimmed from catchphrase"""
        client = BedrockClient()

        mock_response = {
            "body": MagicMock(read=lambda: json.dumps({
                "output": {
                    "message": {
                        "content": [
                            {"text": "\n\n  テスト  \n\n"}
                        ]
                    }
                }
            }).encode())
        }

        with patch.object(client.client, 'invoke_model', return_value=mock_response):
            result = client.generate_catchphrase(background="テスト")

        assert result == "テスト"
        assert not result.startswith(" ")
        assert not result.endswith(" ")
