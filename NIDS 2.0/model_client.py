from __future__ import annotations

import os
import json
import logging
from typing import Any
from dotenv import load_dotenv

from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.wml_client_error import (
    WMLClientError,
    ApiRequestFailure,
)

# Load the .env file
load_dotenv()

logger = logging.getLogger(__name__)
# ===========================================================================
# CONFIGURATION — replace placeholders with your actual values.
# Environment variables override these at runtime if set.
# ===========================================================================

CONFIG: dict[str, str] = {
    # IBM Cloud IAM API key
    "API_KEY": os.environ.get("WML_API_KEY"),

    # Deployment space GUID (Settings → Space GUID in the watsonx.ai console)
    "SPACE_ID": os.environ.get("WML_SPACE_ID"),

    # Deployed model endpoint GUID
    # Extracted from:
    # https://us-south.ml.cloud.ibm.com/ml/v4/deployments/<DEPLOYMENT_ID>/predictions?version=…
    "DEPLOYMENT_ID": os.environ.get("WML_DEPLOYMENT_ID"),

    # Regional WML base URL (no path suffix — the SDK appends its own paths)
    "URL": os.environ.get("WML_URL"),
}

# ===========================================================================
# Label mapping — update to match your deployed model's output classes
# ===========================================================================

LABEL_MAP: dict[int, str] = {
    0: "Normal",
    1: "Anomaly",
}


# ===========================================================================
# ModelClient
# ===========================================================================

class ModelClient:
    """
    Thin wrapper around the IBM watsonx.ai APIClient.

    Instantiation authenticates once and reuses the token for all calls.

    Usage
    -----
        client = ModelClient()          # uses CONFIG values above
        result = client.get_prediction(feature_vector)
    """

    def __init__(
        self,
        api_key:       str = CONFIG["API_KEY"],
        space_id:      str = CONFIG["SPACE_ID"],
        deployment_id: str = CONFIG["DEPLOYMENT_ID"],
        url:           str = CONFIG["URL"],
    ) -> None:
        self._deployment_id = deployment_id

        credentials = Credentials(url=url, api_key=api_key)

        logger.info("Authenticating with IBM watsonx.ai at %s …", url)
        self._client = APIClient(credentials=credentials)
        self._client.set.default_space(space_id)
        logger.info(
            "ModelClient ready (space=%s, deployment=%s)", space_id, deployment_id
        )

    # -----------------------------------------------------------------------
    # Public inference entry point
    # -----------------------------------------------------------------------

    def get_prediction(self, feature_vector: list[Any]) -> dict[str, Any]:
        """
        Send a single feature vector to the deployed model and return a
        structured prediction dictionary.

        Parameters
        ----------
        feature_vector:
            An ordered list of numeric values matching the column order used
            during training.

        Returns
        -------
        On success::

            {
                "prediction":  "Normal" | "Anomaly",
                "raw_label":   <int>,
                "confidence":  <float | None>,
            }

        Raises
        ------
        RuntimeError
            Wraps any WML / network error with a human-readable message so
            the Flask route can return it safely to the caller.
        """
        payload = {
            "input_data": [
                {"values": [feature_vector]}
            ]
        }

        logger.debug("Scoring payload → %s", json.dumps(payload))

        try:
            response = self._client.deployments.score(self._deployment_id, payload)
            logger.debug("Raw WML response ← %s", response)
            return self._parse_response(response)

        # --- Authentication / authorisation failures ----------------------
        except ApiRequestFailure as exc:
            status = getattr(exc, "status_code", None)
            if status == 401:
                msg = (
                    "Authentication failed (HTTP 401). "
                    "Verify that API_KEY in CONFIG is valid and not expired."
                )
            elif status == 403:
                msg = (
                    "Authorisation denied (HTTP 403). "
                    "Check that the API key has 'Editor' access to the deployment space."
                )
            elif status == 404:
                msg = (
                    f"Deployment not found (HTTP 404). "
                    f"Confirm DEPLOYMENT_ID '{self._deployment_id}' is correct "
                    f"and the model is in 'Deployed' state."
                )
            elif status == 422:
                msg = (
                    "Unprocessable payload (HTTP 422). "
                    "The feature vector shape or dtype does not match the model's "
                    "training schema. Check column order and encoding."
                )
            else:
                msg = f"WML API request failed (HTTP {status}): {exc}"

            logger.error(msg)
            raise RuntimeError(msg) from exc

        # --- General WML SDK errors ---------------------------------------
        except WMLClientError as exc:
            msg = f"WML client error: {exc}"
            logger.error(msg)
            raise RuntimeError(msg) from exc

        # --- Malformed / unexpected JSON from the endpoint ----------------
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            msg = (
                f"Malformed response from WML endpoint: {exc}. "
                "The response structure may have changed — check _parse_response()."
            )
            logger.error(msg)
            raise RuntimeError(msg) from exc

        # --- Catch-all: network timeouts, SSL errors, etc. ----------------
        except Exception as exc:
            msg = f"Unexpected error during model scoring: {type(exc).__name__}: {exc}"
            logger.exception(msg)
            raise RuntimeError(msg) from exc

    # Alias so existing callers using .predict() continue to work unchanged
    predict = get_prediction

    # -----------------------------------------------------------------------
    # Internal response parser
    # -----------------------------------------------------------------------

    def _parse_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """
        Extract label and confidence from the standard WML scoring response.

        Expected shape::

            {
              "predictions": [
                {
                  "fields": ["prediction", "probability"],
                  "values": [[0, [0.12, 0.88]]]
                }
              ]
            }
        """
        predictions = response["predictions"][0]
        fields      = predictions["fields"]
        values      = predictions["values"][0]
        result      = dict(zip(fields, values))

        raw_label  = int(result.get("prediction", -1))
        label_text = LABEL_MAP.get(raw_label, "Unknown")

        probability = result.get("probability")
        confidence  = (
            float(max(probability)) if isinstance(probability, list) else None
        )

        return {
            "prediction": label_text,
            "raw_label":  raw_label,
            "confidence": confidence,
        }
