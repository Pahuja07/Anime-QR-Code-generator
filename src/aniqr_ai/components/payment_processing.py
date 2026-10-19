"""Razorpay payment processing and signature verification for AniQR."""

from __future__ import annotations

import hashlib
import hmac
import os
import uuid
from dataclasses import dataclass
from typing import Any
import razorpay

from aniqr_ai.exception import AniQRException
from aniqr_ai.logger import logger

# Credit packages available for purchase
CREDIT_PACKAGES = {
    "pack_starter": {"credits": 20, "amount_inr": 49, "name": "Starter Pack (20 Anime QRs)"},
    "pack_pro": {"credits": 100, "amount_inr": 199, "name": "Pro Creator (100 Anime QRs)"},
    "pack_studio": {"credits": 500, "amount_inr": 799, "name": "Studio Enterprise (500 Anime QRs)"},
}


@dataclass
class RazorpayOrderDetails:
    order_id: str
    amount_inr: int
    amount_subunits: int  # in paise
    currency: str
    credits: int
    customer_token: str
    key_id: str


class RazorpayProcessor:
    """Manages Razorpay order creation, payment signature verification, and webhook handling."""

    def __init__(
        self,
        key_id: str | None = None,
        key_secret: str | None = None,
        webhook_secret: str | None = None,
        is_simulation: bool | None = None,
        allow_simulated_payments: bool | None = None,
    ) -> None:
        self.key_id = key_id or os.getenv("RAZORPAY_KEY_ID", "rzp_test_placeholder_key")
        self.key_secret = key_secret or os.getenv("RAZORPAY_KEY_SECRET", "placeholder_secret")
        self.webhook_secret = webhook_secret or os.getenv("RAZORPAY_WEBHOOK_SECRET", "webhook_secret_key")
        self._placeholder_credentials = any(
            "placeholder" in value.lower()
            for value in (self.key_id, self.key_secret, self.webhook_secret)
        )
        self.allow_simulated_payments = (
            allow_simulated_payments
            if allow_simulated_payments is not None
            else os.getenv("ANIQR_ALLOW_SIMULATED_PAYMENTS", "false").lower() == "true"
        )

        # Auto-detect simulation mode
        if is_simulation is not None:
            self.is_simulation = is_simulation
        else:
            self.is_simulation = (
                "placeholder" in self.key_id
                or "placeholder" in self.key_secret
                or "mock" in self.key_id.lower()
                or self.key_id == "rzp_test_123"
            )

        if not self.is_simulation:
            try:
                self.client = razorpay.Client(auth=(self.key_id, self.key_secret))
            except Exception as e:
                logger.warning(f"Could not initialize Razorpay client ({e}), falling back to simulation.")
                self.client = None
                self.is_simulation = True
        else:
            self.client = None
            logger.info("Razorpay running in simulation/mock mode.")

    def create_order(
        self,
        package_id: str,
        customer_token: str,
        receipt: str | None = None,
    ) -> RazorpayOrderDetails:
        """Create a new Razorpay order for the selected credit package."""
        if package_id not in CREDIT_PACKAGES:
            raise ValueError(
                f"Invalid package_id '{package_id}'. Valid options: {list(CREDIT_PACKAGES.keys())}"
            )

        if self._placeholder_credentials and not self.allow_simulated_payments:
            raise AniQRException(
                "Razorpay credentials are placeholders. Configure real keys or explicitly enable local simulation."
            )

        pkg = CREDIT_PACKAGES[package_id]
        amount_inr = pkg["amount_inr"]
        amount_subunits = amount_inr * 100  # Razorpay expects amounts in paise
        receipt_id = receipt or f"rcpt_{uuid.uuid4().hex[:12]}"

        notes = {
            "package_id": package_id,
            "credits": str(pkg["credits"]),
            "customer_token": customer_token,
        }

        if not self.is_simulation and self.client is not None:
            try:
                order_data = self.client.order.create(
                    {
                        "amount": amount_subunits,
                        "currency": "INR",
                        "receipt": receipt_id,
                        "notes": notes,
                    }
                )
                order_id = order_data["id"]
            except Exception as e:
                logger.error(f"Razorpay order creation failed: {e}")
                raise AniQRException(f"Failed to create Razorpay order: {e}") from e
        else:
            order_id = f"order_{uuid.uuid4().hex[:16]}"
            logger.info(f"Generated simulated Razorpay order: {order_id}")

        return RazorpayOrderDetails(
            order_id=order_id,
            amount_inr=amount_inr,
            amount_subunits=amount_subunits,
            currency="INR",
            credits=pkg["credits"],
            customer_token=customer_token,
            key_id=self.key_id,
        )

    def verify_payment_signature(
        self,
        razorpay_order_id: str,
        razorpay_payment_id: str,
        razorpay_signature: str,
    ) -> bool:
        """Verify the authenticity of the client payment signature using HMAC SHA256."""
        try:
            if self._placeholder_credentials and not self.allow_simulated_payments:
                logger.warning("Payment verification blocked because Razorpay credentials are placeholders")
                return False
            if self.is_simulation and self.allow_simulated_payments:
                return hmac.compare_digest("sim_sig", razorpay_signature)
            # Standard signature verification string: order_id + "|" + payment_id
            payload = f"{razorpay_order_id}|{razorpay_payment_id}"
            expected_signature = hmac.new(
                self.key_secret.encode("utf-8"),
                payload.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()

            # Constant-time comparison
            is_valid = hmac.compare_digest(expected_signature, razorpay_signature)
            if is_valid:
                logger.info(f"Payment signature verified successfully for order: {razorpay_order_id}")
            else:
                logger.warning(f"Payment signature verification failed for order: {razorpay_order_id}")
            return is_valid
        except Exception as e:
            logger.error(f"Error during signature verification: {e}")
            return False

    def verify_webhook_signature(
        self,
        raw_body: str | bytes,
        signature: str,
    ) -> bool:
        """Verify Razorpay webhook event signature against webhook_secret."""
        try:
            if self._placeholder_credentials and not self.allow_simulated_payments:
                return False
            if isinstance(raw_body, str):
                raw_body = raw_body.encode("utf-8")

            expected_signature = hmac.new(
                self.webhook_secret.encode("utf-8"),
                raw_body,
                hashlib.sha256,
            ).hexdigest()

            return hmac.compare_digest(expected_signature, signature)
        except Exception as e:
            logger.error(f"Error verifying webhook signature: {e}")
            return False

    def extract_webhook_payment(
        self, event_payload: dict[str, Any]
    ) -> tuple[str, str, int] | None:
        """Extract (order_id, payment_id, amount_subunits) from a captured payment."""
        event_name = event_payload.get("event")
        if event_name != "payment.captured":
            logger.info(f"Ignoring non-credit webhook event: {event_name}")
            return None

        payment_entity = (
            event_payload.get("payload", {})
            .get("payment", {})
            .get("entity", {})
        )
        order_id = payment_entity.get("order_id")
        payment_id = payment_entity.get("id")
        amount = payment_entity.get("amount")
        if (
            not order_id
            or not payment_id
            or not isinstance(amount, int)
            or payment_entity.get("currency") != "INR"
            or payment_entity.get("status") != "captured"
        ):
            logger.warning("Captured payment webhook is missing valid order or payment details.")
            return None
        return order_id, payment_id, amount
