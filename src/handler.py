import json
import logging
from typing import Any, Dict

from bedrock_client import BedrockClient
from models import CatchphraseRequest, CatchphraseResponse

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda function handler for catchphrase generation

    Args:
        event: API Gateway proxy event
        context: Lambda context

    Returns:
        API Gateway proxy response
    """
    logger.info(f"Received event: {json.dumps(event)}")

    try:
        # Parse request body
        if not event.get("body"):
            return create_response(400, {"error": "Request body is required"})

        body = json.loads(event["body"])
        request = CatchphraseRequest.from_dict(body)

        # Validate request
        is_valid, error_message = request.validate()
        if not is_valid:
            return create_response(400, {"error": error_message})

        # Generate catchphrase
        bedrock_client = BedrockClient()
        catchphrase = bedrock_client.generate_catchphrase(
            starting_word=request.starting_word,
            background=request.background,
            style=request.style
        )

        # Create response
        response = CatchphraseResponse(
            catchphrase=catchphrase,
            model_used=bedrock_client.model_id
        )

        logger.info(f"Generated catchphrase: {catchphrase}")

        return create_response(200, response.to_dict())

    except json.JSONDecodeError:
        return create_response(400, {"error": "Invalid JSON in request body"})
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return create_response(500, {"error": "Internal server error"})


def create_response(status_code: int, body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create API Gateway proxy response

    Args:
        status_code: HTTP status code
        body: Response body dictionary

    Returns:
        API Gateway proxy response
    """
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
            "Access-Control-Allow-Methods": "POST,OPTIONS"
        },
        "body": json.dumps(body, ensure_ascii=False)
    }
