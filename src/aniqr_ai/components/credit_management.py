"""Thread-safe Credit and Guest Token Management for AniQR."""

from __future__ import annotations

import json
import os
import threading
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aniqr_ai.exception import AniQRException
from aniqr_ai.logger import logger


def _get_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class InsufficientCreditsError(AniQRException):
    """Raised when an operation requires more credits than the account possesses."""
    pass


@dataclass
class TransactionRecord:
    transaction_id: str
    type: str  # "grant", "purchase", "deduct", "refund"
    amount: int
    balance_after: int
    timestamp: str
    reference_id: str | None = None
    reason: str | None = None


@dataclass
class UserAccount:
    token: str
    credits: int
    created_at: str
    updated_at: str
    transactions: list[dict[str, Any]] = field(default_factory=list)


class CreditManager:
    """Thread-safe credit ledger with atomic operations and duplicate prevention."""

    _instance: CreditManager | None = None
    _lock = threading.Lock()

    def __init__(self, storage_path: str | Path = "artifacts/credits_ledger.json") -> None:
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._accounts: dict[str, UserAccount] = {}
        self._processed_payment_ids: set[str] = set()
        self._pending_orders: dict[str, dict[str, Any]] = {}
        self._load()

    @classmethod
    def get_instance(cls, storage_path: str | Path = "artifacts/credits_ledger.json") -> CreditManager:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(storage_path)
            return cls._instance

    def _load(self) -> None:
        """Load ledger state from disk."""
        self._accounts.clear()
        self._processed_payment_ids.clear()
        self._pending_orders.clear()
        if not self.storage_path.exists():
            return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                accounts_data = data.get("accounts", {})
                for token, acc in accounts_data.items():
                    self._accounts[token] = UserAccount(
                        token=acc["token"],
                        credits=acc["credits"],
                        created_at=acc["created_at"],
                        updated_at=acc["updated_at"],
                        transactions=acc.get("transactions", []),
                    )
                self._processed_payment_ids = set(data.get("processed_payment_ids", []))
                self._pending_orders = data.get("pending_orders", {})
        except Exception as e:
            raise AniQRException(f"Could not safely load credits ledger {self.storage_path}: {e}") from e

    @contextmanager
    def _guard(self):
        """Serialize updates in this process and refresh the persisted ledger."""
        with self._lock:
            self._load()
            yield

    def _save(self) -> None:
        """Persist ledger state to disk atomically using a temp file."""
        temp_path = self.storage_path.with_suffix(".tmp")
        payload = {
            "accounts": {k: asdict(v) for k, v in self._accounts.items()},
            "processed_payment_ids": list(self._processed_payment_ids),
            "pending_orders": self._pending_orders,
        }
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(temp_path, self.storage_path)

    def create_guest_account(self, initial_credits: int = 3) -> UserAccount:
        """Issue a new secure guest token with initial promotional credits."""
        if not isinstance(initial_credits, int) or isinstance(initial_credits, bool) or initial_credits < 0:
            raise ValueError("initial_credits must be a non-negative integer")
        with self._guard():
            token = f"guest_{uuid.uuid4().hex}"
            now = _get_utc_now()
            tx = TransactionRecord(
                transaction_id=str(uuid.uuid4()),
                type="grant",
                amount=initial_credits,
                balance_after=initial_credits,
                timestamp=now,
                reason="initial_guest_grant",
            )
            account = UserAccount(
                token=token,
                credits=initial_credits,
                created_at=now,
                updated_at=now,
                transactions=[asdict(tx)],
            )
            self._accounts[token] = account
            self._save()
            logger.info(f"Created guest account: token='{token[:12]}...', credits={initial_credits}")
            return account

    def get_account(self, token: str) -> UserAccount | None:
        with self._guard():
            return self._accounts.get(token)

    def get_balance(self, token: str) -> int:
        with self._guard():
            account = self._accounts.get(token)
            if not account:
                raise AniQRException(f"Account with token '{token[:10]}...' not found.")
            return account.credits

    def deduct_credits(
        self,
        token: str,
        amount: int = 1,
        reference_id: str | None = None,
        reason: str = "image_generation",
    ) -> int:
        """Atomically deduct credits. Thread-safe to prevent concurrent double-spending."""
        if not isinstance(amount, int) or isinstance(amount, bool) or amount <= 0:
            raise ValueError("Deduction amount must be a positive integer")
        with self._guard():
            account = self._accounts.get(token)
            if not account:
                raise AniQRException(f"Account with token '{token[:10]}...' not found.")

            if account.credits < amount:
                raise InsufficientCreditsError(
                    f"Insufficient credits: required {amount}, available {account.credits}."
                )

            account.credits -= amount
            now = _get_utc_now()
            account.updated_at = now
            tx = TransactionRecord(
                transaction_id=str(uuid.uuid4()),
                type="deduct",
                amount=amount,
                balance_after=account.credits,
                timestamp=now,
                reference_id=reference_id,
                reason=reason,
            )
            account.transactions.append(asdict(tx))
            self._save()
            logger.info(
                f"Deducted {amount} credit(s) from '{token[:10]}...'. New balance: {account.credits}"
            )
            return account.credits

    def refund_credits(
        self,
        token: str,
        amount: int = 1,
        reference_id: str | None = None,
        reason: str = "generation_failed",
    ) -> int:
        """Refund credits back to user if a generation fails."""
        if not isinstance(amount, int) or isinstance(amount, bool) or amount <= 0:
            raise ValueError("Refund amount must be a positive integer")
        with self._guard():
            account = self._accounts.get(token)
            if not account:
                raise AniQRException(f"Account with token '{token[:10]}...' not found.")

            account.credits += amount
            now = _get_utc_now()
            account.updated_at = now
            tx = TransactionRecord(
                transaction_id=str(uuid.uuid4()),
                type="refund",
                amount=amount,
                balance_after=account.credits,
                timestamp=now,
                reference_id=reference_id,
                reason=reason,
            )
            account.transactions.append(asdict(tx))
            self._save()
            logger.info(
                f"Refunded {amount} credit(s) to '{token[:10]}...'. New balance: {account.credits}"
            )
            return account.credits

    def add_purchased_credits(
        self,
        token: str,
        amount: int,
        payment_id: str,
        order_id: str | None = None,
    ) -> int:
        """Add purchased credits exactly once (strict idempotency protection)."""
        with self._guard():
            if not isinstance(amount, int) or isinstance(amount, bool) or amount <= 0:
                raise ValueError("Purchased credit amount must be a positive integer")
            if not payment_id:
                raise ValueError("payment_id is required")
            account = self._accounts.get(token)
            if not account:
                raise AniQRException(f"Account with token '{token[:10]}...' not found.")

            # Idempotency check: if payment_id already processed, return existing balance
            if payment_id in self._processed_payment_ids:
                logger.warning(
                    f"Duplicate payment detected! Payment ID '{payment_id}' was already credited."
                )
                return account.credits

            if order_id is not None:
                pending = self._pending_orders.get(order_id)
                if (
                    not pending
                    or pending["token"] != token
                    or pending["credits"] != amount
                    or pending.get("paid_payment_id")
                ):
                    raise AniQRException("Payment does not match an unpaid order for this account.")

            account.credits += amount
            self._processed_payment_ids.add(payment_id)
            now = _get_utc_now()
            account.updated_at = now
            tx = TransactionRecord(
                transaction_id=str(uuid.uuid4()),
                type="purchase",
                amount=amount,
                balance_after=account.credits,
                timestamp=now,
                reference_id=payment_id,
                reason=f"razorpay_order_{order_id}",
            )
            account.transactions.append(asdict(tx))
            if order_id is not None:
                self._pending_orders[order_id]["paid_payment_id"] = payment_id
            self._save()
            logger.info(
                f"Credited {amount} purchased credits to '{token[:10]}...' via payment {payment_id}. "
                f"New balance: {account.credits}"
            )
            return account.credits

    def register_pending_order(
        self,
        order_id: str,
        token: str,
        package_id: str,
        amount_subunits: int,
        credits: int,
    ) -> None:
        """Persist server-created order ownership and value for later verification."""
        with self._guard():
            if token not in self._accounts:
                raise AniQRException("Cannot register an order for an unknown account.")
            if (
                not order_id
                or not isinstance(amount_subunits, int)
                or isinstance(amount_subunits, bool)
                or not isinstance(credits, int)
                or isinstance(credits, bool)
                or amount_subunits <= 0
                or credits <= 0
            ):
                raise ValueError("Order id, amount, and credits must be valid")
            record = {
                "token": token,
                "package_id": package_id,
                "amount_subunits": amount_subunits,
                "credits": credits,
            }
            existing = self._pending_orders.get(order_id)
            if existing and any(existing.get(key) != value for key, value in record.items()):
                raise AniQRException("Order id is already registered to a different purchase.")
            self._pending_orders[order_id] = existing or record
            self._save()

    def get_pending_order(self, order_id: str) -> dict[str, Any] | None:
        with self._guard():
            order = self._pending_orders.get(order_id)
            return dict(order) if order else None
