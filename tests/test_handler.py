import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add src directory to Python path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from handler import lambda_handler, create_response


class TestLambdaHandler:
    """Test cases for Lambda handler"""

    def test_create_response(self):
        """Test create_response function"""
        response = create_response(200, {"message": "success"})

        assert response["statusCode"] == 200
        assert "Content-Type" in response["headers"]
        assert "Access-Control-Allow-Origin" in response["headers"]
        body = json.loads(response["body"])
        assert body["message"] == "success"

    def test_missing_body(self):
        """Test handler with missing request body"""
        event = {}
        context = MagicMock()

        response = lambda_handler(event, context)

        assert response["statusCode"] == 400
        body = json.loads(response["body"])
        assert "error" in body

    def test_invalid_json(self):
        """Test handler with invalid JSON"""
        event = {"body": "invalid json"}
        context = MagicMock()

        response = lambda_handler(event, context)

        assert response["statusCode"] == 400
        body = json.loads(response["body"])
        assert "error" in body

    def test_missing_background(self):
        """Test handler with missing required background field"""
        event = {
            "body": json.dumps({
                "starting_word": "未来",
                "style": "前向き"
            })
        }
        context = MagicMock()

        response = lambda_handler(event, context)

        assert response["statusCode"] == 400
        body = json.loads(response["body"])
        assert "error" in body
        assert "background" in body["error"]

    @patch("handler.BedrockClient")
    def test_successful_generation(self, mock_bedrock_class):
        """Test successful catchphrase generation"""
        # Setup mock
        mock_client = MagicMock()
        mock_client.generate_catchphrase.return_value = "未来を創る、今日から始まる"
        mock_client.model_id = "us.amazon.nova-micro-v1:0"
        mock_bedrock_class.return_value = mock_client

        event = {
            "body": json.dumps({
                "starting_word": "未来",
                "background": "テクノロジー企業",
                "style": "前向き"
            })
        }
        context = MagicMock()

        response = lambda_handler(event, context)

        assert response["statusCode"] == 200
        body = json.loads(response["body"])
        assert "catchphrase" in body
        assert body["catchphrase"] == "未来を創る、今日から始まる"
        assert "model_used" in body

        # Verify Bedrock client was called correctly
        mock_client.generate_catchphrase.assert_called_once_with(
            starting_word="未来",
            background="テクノロジー企業",
            style="前向き"
        )

    @patch("handler.BedrockClient")
    def test_bedrock_error(self, mock_bedrock_class):
        """Test handler when Bedrock raises an error"""
        # Setup mock to raise exception
        mock_client = MagicMock()
        mock_client.generate_catchphrase.side_effect = Exception("Bedrock error")
        mock_bedrock_class.return_value = mock_client

        event = {
            "body": json.dumps({
                "background": "テクノロジー企業"
            })
        }
        context = MagicMock()

        response = lambda_handler(event, context)

        assert response["statusCode"] == 500
        body = json.loads(response["body"])
        assert "error" in body


class TestModels:
    """Test cases for models"""

    def test_catchphrase_request_from_dict(self):
        """Test CatchphraseRequest.from_dict"""
        from models import CatchphraseRequest

        data = {
            "starting_word": "未来",
            "background": "テクノロジー",
            "style": "前向き"
        }
        request = CatchphraseRequest.from_dict(data)

        assert request.starting_word == "未来"
        assert request.background == "テクノロジー"
        assert request.style == "前向き"

    def test_catchphrase_request_validation(self):
        """Test CatchphraseRequest validation"""
        from models import CatchphraseRequest

        # Valid request
        request = CatchphraseRequest(background="テスト")
        is_valid, error = request.validate()
        assert is_valid is True
        assert error is None

        # Invalid request (missing background)
        request = CatchphraseRequest()
        is_valid, error = request.validate()
        assert is_valid is False
        assert error is not None

    def test_catchphrase_response_to_dict(self):
        """Test CatchphraseResponse.to_dict"""
        from models import CatchphraseResponse

        response = CatchphraseResponse(
            catchphrase="テストフレーズ",
            model_used="test-model"
        )
        result = response.to_dict()

        assert result["catchphrase"] == "テストフレーズ"
        assert result["model_used"] == "test-model"
