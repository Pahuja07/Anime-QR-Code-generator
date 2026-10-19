from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pytest

from aniqr_ai.components.credit_management import (
    CreditManager,
    InsufficientCreditsError,
)


def test_guest_account_creation_and_balance(tmp_path: Path):
    ledger_file = tmp_path / "test_credits.json"
    manager = CreditManager(storage_path=ledger_file)

    account = manager.create_guest_account(initial_credits=5)
    assert account.token.startswith("guest_")
    assert account.credits == 5
    assert manager.get_balance(account.token) == 5


def test_credit_deduction_and_insufficient_funds(tmp_path: Path):
    ledger_file = tmp_path / "test_credits.json"
    manager = CreditManager(storage_path=ledger_file)

    account = manager.create_guest_account(initial_credits=2)
    new_balance = manager.deduct_credits(account.token, amount=1)
    assert new_balance == 1

    new_balance = manager.deduct_credits(account.token, amount=1)
    assert new_balance == 0

    with pytest.raises(InsufficientCreditsError):
        manager.deduct_credits(account.token, amount=1)


def test_credit_refund(tmp_path: Path):
    ledger_file = tmp_path / "test_credits.json"
    manager = CreditManager(storage_path=ledger_file)

    account = manager.create_guest_account(initial_credits=3)
    manager.deduct_credits(account.token, amount=2)
    assert manager.get_balance(account.token) == 1

    refunded_balance = manager.refund_credits(account.token, amount=2, reason="generation_failed")
    assert refunded_balance == 3
    assert manager.get_balance(account.token) == 3


def test_payment_idempotency_prevents_duplicate_credits(tmp_path: Path):
    ledger_file = tmp_path / "test_credits.json"
    manager = CreditManager(storage_path=ledger_file)

    account = manager.create_guest_account(initial_credits=0)
    payment_id = "pay_test_unique_12345"

    # First credit event
    bal1 = manager.add_purchased_credits(account.token, amount=50, payment_id=payment_id)
    assert bal1 == 50

    # Duplicate webhook / event with same payment_id
    bal2 = manager.add_purchased_credits(account.token, amount=50, payment_id=payment_id)
    assert bal2 == 50
    assert manager.get_balance(account.token) == 50


def test_concurrent_deduction_prevents_double_spending(tmp_path: Path):
    ledger_file = tmp_path / "test_credits.json"
    manager = CreditManager(storage_path=ledger_file)

    # 10 credits in total
    account = manager.create_guest_account(initial_credits=10)

    # Launch 25 concurrent deduction attempts of 1 credit each
    # Exactly 10 should succeed, and 15 should raise InsufficientCreditsError
    success_count = 0
    failure_count = 0

    def try_deduct():
        try:
            manager.deduct_credits(account.token, amount=1)
            return True
        except InsufficientCreditsError:
            return False

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: try_deduct(), range(25)))

    success_count = sum(1 for r in results if r)
    failure_count = sum(1 for r in results if not r)

    assert success_count == 10
    assert failure_count == 15
    assert manager.get_balance(account.token) == 0
