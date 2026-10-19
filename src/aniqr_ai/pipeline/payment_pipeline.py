"""Payment and credit fulfillment pipeline for AniQR."""

from __future__ import annotations

from typing import Any, Callable
from aniqr_ai.components.credit_management import CreditManager, InsufficientCreditsError
from aniqr_ai.components.payment_processing import (
    CREDIT_PACKAGES,
    RazorpayOrderDetails,
    RazorpayProcessor,
)
from aniqr_ai.exception import AniQRException
from aniqr_ai.logger import logger


class PaymentPipeline:
    """Manages order creation, signature verification, credit fulfillment, and auto-refunds."""

    def __init__(
        self,
        credit_manager: CreditManager | None = None,
        processor: RazorpayProcessor | None = None,
    ) -> None:
        self.credit_manager = credit_manager or CreditManager.get_instance()
        self.processor = processor or RazorpayProcessor()

    def create_checkout_order(self, token: str, package_id: str) -> RazorpayOrderDetails:
        """Create a payment order for a user token."""
        # Ensure account exists
        account = self.credit_manager.get_account(token)
        if not account:
            raise AniQRException(f"Account with token '{token}' not found.")
        order = self.processor.create_order(package_id=package_id, customer_token=token)
        self.credit_manager.register_pending_order(
            order_id=order.order_id,
            token=token,
            package_id=package_id,
            amount_subunits=order.amount_subunits,
            credits=order.credits,
        )
        return order

    def fulfill_payment(
        self,
        token: str,
        order_id: str,
        payment_id: str,
        signature: str,
        credits: int,
        package_id: str | None = None,
    ) -> int:
        """Verify payment signature and credit account idempotently."""
        if self.processor.is_simulation and not self.processor.allow_simulated_payments:
            raise AniQRException("Simulated payments are disabled.")
        is_valid = self.processor.verify_payment_signature(
            razorpay_order_id=order_id,
            razorpay_payment_id=payment_id,
            razorpay_signature=signature,
        )
        if not is_valid:
            raise AniQRException("Payment verification failed: Invalid Razorpay signature.")

        pending_order = self.credit_manager.get_pending_order(order_id)
        if (
            not pending_order
            or pending_order["token"] != token
            or pending_order["credits"] != credits
            or (package_id is not None and pending_order["package_id"] != package_id)
        ):
            raise AniQRException("Payment does not match an order for this account and package.")

        if not self.processor.is_simulation:
            try:
                payment = self.processor.client.payment.fetch(payment_id)
            except Exception as exc:
                raise AniQRException("Could not confirm payment status with Razorpay.") from exc
            if (
                payment.get("order_id") != order_id
                or payment.get("status") != "captured"
                or payment.get("amount") != pending_order["amount_subunits"]
                or payment.get("currency") != "INR"
            ):
                raise AniQRException("Razorpay payment is not captured for the expected order amount.")

        new_balance = self.credit_manager.add_purchased_credits(
            token=token,
            amount=credits,
            payment_id=payment_id,
            order_id=order_id,
        )
        logger.info(f"Payment fulfilled: token='{token[:10]}...', added={credits}, new_balance={new_balance}")
        return new_balance

    def fulfill_webhook(
        self,
        raw_body: str | bytes,
        signature: str,
        event_payload: dict[str, Any],
    ) -> bool:
        """Process and verify incoming Razorpay webhook."""
        if not self.processor.verify_webhook_signature(raw_body, signature):
            logger.warning("Webhook rejected: Invalid signature.")
            return False

        extracted = self.processor.extract_webhook_payment(event_payload)
        if not extracted:
            return True  # Event recognized but requires no credit action

        order_id, payment_id, amount_subunits = extracted
        pending_order = self.credit_manager.get_pending_order(order_id)
        if (
            not pending_order
            or amount_subunits != pending_order["amount_subunits"]
        ):
            logger.error("Captured-payment webhook did not match a pending order")
            return False
        self.credit_manager.add_purchased_credits(
            token=pending_order["token"],
            amount=pending_order["credits"],
            payment_id=payment_id,
            order_id=order_id,
        )
        return True

    def execute_with_credits(
        self,
        token: str,
        cost: int,
        generation_func: Callable[[], Any],
    ) -> tuple[Any, int]:
        """Atomically deduct credit, run generation, and auto-refund on failure."""
        self.credit_manager.deduct_credits(token=token, amount=cost, reason="anime_qr_generation")

        try:
            result = generation_func()
            remaining_balance = self.credit_manager.get_balance(token)
            return result, remaining_balance
        except Exception as e:
            logger.warning(
                f"Generation failed ({e}). Automatically refunding {cost} credit(s) to '{token[:10]}...'"
            )
            self.credit_manager.refund_credits(token=token, amount=cost, reason="generation_error")
            raise
