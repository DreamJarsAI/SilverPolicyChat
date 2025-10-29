"""Authentication and user management helpers for the Gradio interface."""
from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import logging
import re
import secrets
import smtplib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from typing import Any, Dict, Tuple

from sqlalchemy import select

from config import Settings
from policy_store import PolicyStore, UserAccount


logger = logging.getLogger(__name__)


class SecretHasher:
    """Hash and verify secrets using PBKDF2-HMAC."""

    algorithm = "pbkdf2_sha256"
    iterations = 310_000

    def hash_secret(self, secret: str) -> str:
        salt = secrets.token_bytes(16)
        derived = hashlib.pbkdf2_hmac(
            "sha256", secret.encode("utf-8"), salt, self.iterations
        )
        payload = "$".join(
            (
                self.algorithm,
                str(self.iterations),
                base64.b64encode(salt).decode("utf-8"),
                base64.b64encode(derived).decode("utf-8"),
            )
        )
        return payload

    def verify_secret(self, secret: str, encoded: str) -> bool:
        try:
            algorithm, iteration_s, salt_b64, derived_b64 = encoded.split("$")
        except ValueError:
            logger.warning("Invalid secret hash format encountered.")
            return False

        if algorithm != self.algorithm:
            logger.warning("Unsupported hashing algorithm: %s", algorithm)
            return False

        try:
            iterations = int(iteration_s)
        except ValueError:
            logger.warning("Invalid iteration count in stored hash.")
            return False

        try:
            salt = base64.b64decode(salt_b64)
            expected = base64.b64decode(derived_b64)
        except (ValueError, binascii.Error):
            logger.warning("Failed to decode stored hash components.")
            return False

        candidate = hashlib.pbkdf2_hmac(
            "sha256", secret.encode("utf-8"), salt, iterations
        )
        return hmac.compare_digest(candidate, expected)


@dataclass
class EmailSettings:
    host: str | None
    port: int | None
    username: str | None
    password: str | None
    from_email: str | None
    use_tls: bool
    use_ssl: bool
    dev_mode: bool

    @classmethod
    def from_settings(cls, settings: Settings) -> "EmailSettings":
        host = getattr(settings, "smtp_host", None)
        port = getattr(settings, "smtp_port", None)
        username = getattr(settings, "smtp_username", None)
        password = getattr(settings, "smtp_password", None)
        from_email = getattr(settings, "smtp_from_email", None)
        use_tls = getattr(settings, "smtp_use_tls", False)
        use_ssl = getattr(settings, "smtp_use_ssl", False)
        dev_mode = getattr(settings, "smtp_dev_mode", False)
        if host is None:
            dev_mode = True
        return cls(
            host=host,
            port=port,
            username=username,
            password=password,
            from_email=from_email,
            use_tls=use_tls,
            use_ssl=use_ssl,
            dev_mode=dev_mode,
        )


class EmailSender:
    """Send transactional emails with graceful development fallbacks."""

    def __init__(self, settings: EmailSettings) -> None:
        self.settings = settings

    def send_verification_code(self, email: str, code: str, *, username: str) -> None:
        subject = "Verify your School Policy Assistant account"
        body = (
            "Hello {username},\n\n"
            "Use the following verification code to finish setting up your account: {code}.\n"
            "The code expires in 15 minutes.\n\n"
            "If you did not request this, please ignore this email."
        ).format(username=username, code=code)
        self._send(email, subject, body)

    def send_reset_code(self, email: str, code: str, *, username: str) -> None:
        subject = "Reset your School Policy Assistant password"
        body = (
            "Hello {username},\n\n"
            "A password or username reset was requested for your account.\n"
            "Use this verification code to proceed: {code}.\n"
            "The code expires in 15 minutes.\n\n"
            "If you did not request this change, contact support immediately."
        ).format(username=username, code=code)
        self._send(email, subject, body)

    def _send(self, recipient: str, subject: str, body: str) -> None:
        if self.settings.dev_mode:
            logger.info(
                "SMTP disabled or in dev mode — simulated email to %s with subject %s. Body: %s",
                recipient,
                subject,
                body,
            )
            return

        if not self.settings.host:
            raise RuntimeError("SMTP host is not configured; cannot send email.")

        port = self.settings.port or (465 if self.settings.use_ssl else 587)
        from_email = self.settings.from_email or self.settings.username
        if not from_email:
            raise RuntimeError("SMTP_FROM_EMAIL or SMTP_USERNAME must be configured for sending email.")

        message = EmailMessage()
        message["To"] = recipient
        message["From"] = from_email
        message["Subject"] = subject
        message.set_content(body)

        if self.settings.use_ssl:
            smtp_cls = smtplib.SMTP_SSL
        else:
            smtp_cls = smtplib.SMTP

        with smtp_cls(self.settings.host, port, timeout=30) as client:
            if self.settings.use_tls and not self.settings.use_ssl:
                client.starttls()
            if self.settings.username and self.settings.password:
                client.login(self.settings.username, self.settings.password)
            client.send_message(message)


def generate_verification_code() -> str:
    return f"{secrets.randbelow(1_000_000):06d}"


USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]{3,32}$")
PASSWORD_SPECIAL_PATTERN = re.compile(r"[^A-Za-z0-9]")


def validate_username(username: str) -> Tuple[bool, str]:
    if not username:
        return False, "Username is required."
    if not USERNAME_PATTERN.fullmatch(username):
        return (
            False,
            "Username must be 3-32 characters and can only include letters, numbers, periods, underscores, or dashes.",
        )
    return True, ""


def validate_password(password: str) -> Tuple[bool, str]:
    if len(password) < 12:
        return False, "Password must be at least 12 characters long."
    if password.lower() == password or password.upper() == password:
        return False, "Password must include a mix of uppercase and lowercase letters."
    if not any(ch.isdigit() for ch in password):
        return False, "Password must include at least one number."
    if not PASSWORD_SPECIAL_PATTERN.search(password):
        return False, "Password must include at least one special character."
    return True, ""


def validate_email(email: str) -> Tuple[bool, str]:
    if not email:
        return False, "Email is required."
    email_lower = email.lower()
    if not email_lower.endswith("@nyu.edu"):
        return False, "A valid NYU email ending with @nyu.edu is required."
    return True, ""


def _as_utc(timestamp: datetime) -> datetime:
    """Normalize potentially naive timestamps (e.g. from SQLite) to UTC-aware datetimes."""
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


class AuthService:
    """Encapsulates user registration, authentication, and password reset flows."""

    verification_ttl = timedelta(minutes=15)

    def __init__(self, store: PolicyStore, email_sender: EmailSender, hasher: SecretHasher) -> None:
        self.store = store
        self.email_sender = email_sender
        self.hasher = hasher

    def initiate_registration(self, username: str, email: str) -> Tuple[bool, str, Dict[str, Any]]:
        username_normalized = username.strip().lower()
        email_normalized = email.strip().lower()

        valid_username, username_msg = validate_username(username_normalized)
        if not valid_username:
            return False, username_msg, {}

        valid_email, email_msg = validate_email(email_normalized)
        if not valid_email:
            return False, email_msg, {}

        with self.store.session() as session:
            existing_username = session.execute(
                select(UserAccount).where(UserAccount.username == username_normalized)
            ).scalar_one_or_none()
            existing_email = session.execute(
                select(UserAccount).where(UserAccount.email == email_normalized)
            ).scalar_one_or_none()

            if existing_username and existing_username.email != email_normalized:
                return False, "Username is already in use by another account.", {}

            if existing_email and existing_email.username != username_normalized:
                return False, "Email is already registered; try logging in or resetting your password.", {}

            if existing_username and existing_username.is_verified:
                return False, "Account already exists. Please log in instead.", {}

            if existing_email and existing_email.is_verified:
                return False, "Email already verified. Use the login form or reset your password.", {}

            code = generate_verification_code()
            hashed_code = self.hasher.hash_secret(code)
            now = datetime.now(timezone.utc)

            if existing_username:
                user = existing_username
                user.email = email_normalized
                user.verification_hash = hashed_code
                user.verification_sent_at = now
                user.is_verified = False
            elif existing_email:
                user = existing_email
                user.username = username_normalized
                user.verification_hash = hashed_code
                user.verification_sent_at = now
                user.is_verified = False
            else:
                user = UserAccount(
                    username=username_normalized,
                    email=email_normalized,
                    password_hash=None,
                    is_verified=False,
                    verification_hash=hashed_code,
                    verification_sent_at=now,
                )
                session.add(user)

            session.flush()

        try:
            self.email_sender.send_verification_code(email_normalized, code, username=username_normalized)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to send verification email: %s", exc)
            return (
                False,
                "Unable to send verification email. Please try again later or contact support.",
                {},
            )

        return True, "Verification code sent. Check your NYU email and enter the 6-digit code.", {
            "username": username_normalized,
            "email": email_normalized,
        }

    def complete_registration(self, state: Dict[str, Any], code: str, password: str) -> Tuple[bool, str]:
        username = state.get("username")
        email = state.get("email")
        if not username or not email:
            return False, "Start the registration process before entering the verification code."

        with self.store.session() as session:
            user = session.execute(
                select(UserAccount).where(UserAccount.username == username)
            ).scalar_one_or_none()
            if not user or user.email != email:
                return False, "Registration session not found. Please restart the sign-up process."

            if not user.verification_hash or not user.verification_sent_at:
                return False, "Request a new verification code to continue registration."

            sent_at = _as_utc(user.verification_sent_at)
            if datetime.now(timezone.utc) - sent_at > self.verification_ttl:
                return False, "Verification code expired. Request a new one."

            code_normalized = code.strip()
            if not code_normalized:
                return False, "Enter the 6-digit verification code that was emailed to you."

            if not self.hasher.verify_secret(code_normalized, user.verification_hash):
                return False, "Invalid verification code."

            strong, message = validate_password(password)
            if not strong:
                return False, message

            user.password_hash = self.hasher.hash_secret(password)
            user.is_verified = True
            user.verification_hash = None
            user.verification_sent_at = None
            user.reset_code_hash = None
            user.reset_requested_at = None
            session.flush()

        return True, "Registration complete. You can now log in with your username and password."

    def authenticate(self, username: str, password: str) -> Tuple[bool, str, Dict[str, Any] | None]:
        username_normalized = username.strip().lower()
        password = password or ""

        with self.store.session() as session:
            user = session.execute(
                select(UserAccount).where(UserAccount.username == username_normalized)
            ).scalar_one_or_none()

            if not user:
                return False, "Account not found. Check your username or register for a new account.", None

            if not user.is_verified:
                return False, "Account not verified. Complete registration before logging in.", None

            if not user.password_hash or not self.hasher.verify_secret(password, user.password_hash):
                return False, "Incorrect password. Try again or reset your password.", None

        return True, "Login successful.", {
            "username": username_normalized,
            "email": user.email,
        }

    def initiate_password_reset(self, email: str) -> Tuple[bool, str, Dict[str, Any]]:
        email_normalized = email.strip().lower()
        valid_email, email_msg = validate_email(email_normalized)
        if not valid_email:
            return False, email_msg, {}

        with self.store.session() as session:
            user = session.execute(
                select(UserAccount).where(UserAccount.email == email_normalized)
            ).scalar_one_or_none()

            if not user or not user.is_verified:
                return False, "No verified account found for that email. Register first if you're new.", {}

            code = generate_verification_code()
            user.reset_code_hash = self.hasher.hash_secret(code)
            user.reset_requested_at = datetime.now(timezone.utc)
            session.flush()
            username = user.username

        try:
            self.email_sender.send_reset_code(email_normalized, code, username=username)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to send reset email: %s", exc)
            return False, "Unable to send reset email right now. Try again later.", {}

        return True, "Password reset email sent. Enter the 6-digit code from your inbox.", {
            "email": email_normalized,
        }

    def complete_password_reset(self, state: Dict[str, Any], code: str, password: str) -> Tuple[bool, str]:
        email = state.get("email")
        if not email:
            return False, "Start the reset process before entering the verification code."

        with self.store.session() as session:
            user = session.execute(select(UserAccount).where(UserAccount.email == email)).scalar_one_or_none()
            if not user or not user.reset_code_hash or not user.reset_requested_at:
                return False, "Request a new reset code to continue."

            requested_at = _as_utc(user.reset_requested_at)
            if datetime.now(timezone.utc) - requested_at > self.verification_ttl:
                return False, "Reset code expired. Request a new one."

            code_normalized = code.strip()
            if not code_normalized:
                return False, "Enter the 6-digit verification code sent to your email."

            if not self.hasher.verify_secret(code_normalized, user.reset_code_hash):
                return False, "Invalid verification code."

            strong, message = validate_password(password)
            if not strong:
                return False, message

            user.password_hash = self.hasher.hash_secret(password)
            user.reset_code_hash = None
            user.reset_requested_at = None
            session.flush()

        return True, "Password updated. You can now log in with your new credentials."

