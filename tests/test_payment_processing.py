import hashlib
import hmac
import json
from pathlib import Path
import pytest

from aniqr_ai.components.credit_management import CreditManager
from aniqr_ai.components.payment_processing import (
    CREDIT_PACKAGES,
    RazorpayProcessor,
)
from aniqr_ai.pipeline.payment_pipeline import PaymentPipeline
from aniqr_ai.exception import AniQRException


def test_order_creation():
    processor = RazorpayProcessor(key_id="rzp_test_123", key_secret="secret_abc")
    order = processor.create_order(
        package_id="pack_pro",
        customer_token="guest_abc123",
    )
    assert order.amount_inr == 199
    assert order.amount_subunits == 19900
    assert order.credits == 100
    assert order.order_id.startswith("order_")


def test_signature_verification_success_and_failure():
    secret = "my_secret_key_999"
    processor = RazorpayProcessor(key_id="rzp_test_123", key_secret=secret)

    order_id = "order_xyz890"
    payment_id = "pay_abc567"

    # Compute valid signature
    payload = f"{order_id}|{payment_id}"
    valid_signature = hmac.new(
        secret.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    assert processor.verify_payment_signature(order_id, payment_id, valid_signature)
    assert not processor.verify_payment_signature(order_id, payment_id, "invalid_signature_hex")


def test_webhook_verification_and_fulfillment(tmp_path: Path):
    ledger_file = tmp_path / "test_credits_webhook.json"
    credit_mgr = CreditManager(storage_path=ledger_file)
    account = credit_mgr.create_guest_account(initial_credits=0)
    order_id = "order_hook_123456"
    credit_mgr.register_pending_order(
        order_id=order_id,
        token=account.token,
        package_id="pack_starter",
        amount_subunits=4900,
        credits=20,
    )

    webhook_secret = "whsec_supersecret123"
    processor = RazorpayProcessor(
        key_id="rzp_test_123",
        key_secret="secret_abc",
        webhook_secret=webhook_secret,
    )
    pipeline = PaymentPipeline(credit_manager=credit_mgr, processor=processor)

    # Simulated Razorpay payment.captured webhook payload
    webhook_data = {
        "event": "payment.captured",
        "payload": {
            "payment": {
                "entity": {
                    "id": "pay_hook_987654",
                    "order_id": order_id,
                    "amount": 4900,
                    "currency": "INR",
                    "status": "captured",
                }
            }
        },
    }
    raw_body = json.dumps(webhook_data)
    valid_wh_sig = hmac.new(
        webhook_secret.encode("utf-8"),
        raw_body.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    # Process valid webhook
    success = pipeline.fulfill_webhook(raw_body, valid_wh_sig, webhook_data)
    assert success
    assert credit_mgr.get_balance(account.token) == 20

    # Ensure idempotency if webhook is delivered twice
    success_retry = pipeline.fulfill_webhook(raw_body, valid_wh_sig, webhook_data)
    assert success_retry
    assert credit_mgr.get_balance(account.token) == 20


def test_generation_with_credit_auto_refund(tmp_path: Path):
    ledger_file = tmp_path / "test_credits_refund.json"
    credit_mgr = CreditManager(storage_path=ledger_file)
    account = credit_mgr.create_guest_account(initial_credits=5)
    pipeline = PaymentPipeline(credit_manager=credit_mgr)

    # 1. Successful generation: 1 credit deducted
    result, remaining = pipeline.execute_with_credits(
        token=account.token,
        cost=1,
        generation_func=lambda: "image_data_mock",
    )
    assert result == "image_data_mock"
    assert remaining == 4
    assert credit_mgr.get_balance(account.token) == 4

    # A failed generation refunds the deducted credit.
    def failing_generation():
        raise RuntimeError("GPU Out of Memory")

    with pytest.raises(RuntimeError):
        pipeline.execute_with_credits(
            token=account.token,
            cost=1,
            generation_func=failing_generation,
        )

    assert credit_mgr.get_balance(account.token) == 4


def test_payment_requires_matching_registered_order_and_package(tmp_path: Path):
    credit_mgr = CreditManager(storage_path=tmp_path / "orders.json")
    owner = credit_mgr.create_guest_account(initial_credits=0)
    other_user = credit_mgr.create_guest_account(initial_credits=0)
    processor = RazorpayProcessor(
        key_id="mock_key",
        key_secret="mock_secret",
        is_simulation=True,
        allow_simulated_payments=True,
    )
    pipeline = PaymentPipeline(credit_manager=credit_mgr, processor=processor)
    order = pipeline.create_checkout_order(owner.token, "pack_starter")

    with pytest.raises(AniQRException, match="does not match an order"):
        pipeline.fulfill_payment(
            token=other_user.token,
            order_id=order.order_id,
            payment_id="pay_mock_1",
            signature="sim_sig",
            credits=20,
            package_id="pack_starter",
        )
    with pytest.raises(AniQRException, match="does not match an order"):
        pipeline.fulfill_payment(
            token=owner.token,
            order_id=order.order_id,
            payment_id="pay_mock_2",
            signature="sim_sig",
            credits=500,
            package_id="pack_studio",
        )

    assert pipeline.fulfill_payment(
        token=owner.token,
        order_id=order.order_id,
        payment_id="pay_mock_3",
        signature="sim_sig",
        credits=20,
        package_id="pack_starter",
    ) == 20
    assert credit_mgr.get_balance(owner.token) == 20
