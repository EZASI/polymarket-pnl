#!/usr/bin/env python3
"""
SECURITY MODULE — AES-256 encryption for private keys + utilities
==================================================================

Encrypts .env file so private keys never sit as plaintext on disk.

Usage:
  # First time: encrypt your .env file
  python3 security.py encrypt

  # To decrypt (bot does this automatically on startup)
  python3 security.py decrypt

  # To change password
  python3 security.py rekey
"""

import base64
import getpass
import hashlib
import json
import os
import secrets
import sys
from pathlib import Path


def _derive_key(password: str, salt: bytes) -> bytes:
    """Derive AES-256 key from password using PBKDF2."""
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations=100_000)


def encrypt_file(plaintext_path: str, encrypted_path: str, password: str = None):
    """Encrypt a file with AES-256-CBC using a password."""
    try:
        from Crypto.Cipher import AES
        from Crypto.Util.Padding import pad
    except ImportError:
        # Fallback: use simpler XOR-based encryption with PBKDF2
        return _encrypt_simple(plaintext_path, encrypted_path, password)

    if password is None:
        password = getpass.getpass("Set encryption password: ")
        confirm = getpass.getpass("Confirm password: ")
        if password != confirm:
            print("ERROR: Passwords don't match")
            sys.exit(1)

    data = open(plaintext_path, "rb").read()
    salt = secrets.token_bytes(16)
    key = _derive_key(password, salt)
    iv = secrets.token_bytes(16)

    cipher = AES.new(key, AES.MODE_CBC, iv)
    encrypted = cipher.encrypt(pad(data, AES.block_size))

    # Store: salt (16) + iv (16) + encrypted data
    with open(encrypted_path, "wb") as f:
        f.write(salt + iv + encrypted)

    print(f"Encrypted {plaintext_path} -> {encrypted_path}")
    return True


def decrypt_file(encrypted_path: str, password: str = None) -> str:
    """Decrypt a file and return contents as string."""
    try:
        from Crypto.Cipher import AES
        from Crypto.Util.Padding import unpad
    except ImportError:
        return _decrypt_simple(encrypted_path, password)

    if password is None:
        password = getpass.getpass("Decryption password: ")

    data = open(encrypted_path, "rb").read()
    salt = data[:16]
    iv = data[16:32]
    encrypted = data[32:]

    key = _derive_key(password, salt)
    cipher = AES.new(key, AES.MODE_CBC, iv)

    try:
        decrypted = unpad(cipher.decrypt(encrypted), AES.block_size)
        return decrypted.decode()
    except (ValueError, KeyError):
        raise ValueError("Wrong password or corrupted file")


def _encrypt_simple(plaintext_path: str, encrypted_path: str, password: str = None):
    """Fallback encryption using XOR + PBKDF2 (no pycryptodome needed)."""
    if password is None:
        password = getpass.getpass("Set encryption password: ")
        confirm = getpass.getpass("Confirm password: ")
        if password != confirm:
            print("ERROR: Passwords don't match")
            sys.exit(1)

    data = open(plaintext_path, "rb").read()
    salt = secrets.token_bytes(32)
    key = _derive_key(password, salt)

    # XOR encryption with key-derived stream
    stream = bytearray()
    for i in range(0, len(data), 32):
        block_key = hashlib.sha256(key + i.to_bytes(4, "big")).digest()
        stream.extend(block_key)
    stream = stream[:len(data)]

    encrypted = bytes(a ^ b for a, b in zip(data, stream))

    payload = {
        "salt": base64.b64encode(salt).decode(),
        "data": base64.b64encode(encrypted).decode(),
        "version": 1,
    }

    with open(encrypted_path, "w") as f:
        json.dump(payload, f)

    print(f"Encrypted {plaintext_path} -> {encrypted_path}")
    return True


def _decrypt_simple(encrypted_path: str, password: str = None) -> str:
    """Fallback decryption."""
    if password is None:
        password = getpass.getpass("Decryption password: ")

    payload = json.loads(open(encrypted_path).read())
    salt = base64.b64decode(payload["salt"])
    encrypted = base64.b64decode(payload["data"])

    key = _derive_key(password, salt)

    stream = bytearray()
    for i in range(0, len(encrypted), 32):
        block_key = hashlib.sha256(key + i.to_bytes(4, "big")).digest()
        stream.extend(block_key)
    stream = stream[:len(encrypted)]

    decrypted = bytes(a ^ b for a, b in zip(encrypted, stream))

    try:
        return decrypted.decode()
    except UnicodeDecodeError:
        raise ValueError("Wrong password or corrupted file")


def load_encrypted_env(encrypted_path: str = ".env.enc", password: str = None) -> dict:
    """Load and parse encrypted .env file into a dict."""
    if not Path(encrypted_path).exists():
        return {}

    contents = decrypt_file(encrypted_path, password)
    env = {}
    for line in contents.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip()
    return env


def setup_encrypted_env():
    """Interactive: encrypt .env -> .env.enc, then delete plaintext."""
    env_path = Path(".env")
    enc_path = Path(".env.enc")

    if not env_path.exists():
        print("ERROR: .env file not found. Create it first:")
        print("  cp .env.example .env")
        print("  # Edit .env with your private key")
        sys.exit(1)

    if enc_path.exists():
        overwrite = input(".env.enc already exists. Overwrite? (y/N): ")
        if overwrite.lower() != "y":
            print("Aborted.")
            return

    print("\nThis will encrypt your .env file with a password.")
    print("You'll need this password every time the bot starts.\n")

    encrypt_file(str(env_path), str(enc_path))

    delete = input("\nDelete plaintext .env? (recommended) (y/N): ")
    if delete.lower() == "y":
        env_path.unlink()
        print("Plaintext .env deleted. Only .env.enc remains.")
    else:
        print("WARNING: Plaintext .env still exists. Delete it manually for security.")

    print("\nDone. The bot will prompt for your password on startup.")


# ============================================================
# CLI
# ============================================================
def main():
    if len(sys.argv) < 2:
        print("Usage: python3 security.py [encrypt|decrypt|rekey]")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "encrypt":
        setup_encrypted_env()
    elif cmd == "decrypt":
        if not Path(".env.enc").exists():
            print("No .env.enc file found.")
            sys.exit(1)
        contents = decrypt_file(".env.enc")
        print(contents)
    elif cmd == "rekey":
        if not Path(".env.enc").exists():
            print("No .env.enc file found. Run 'encrypt' first.")
            sys.exit(1)
        print("Enter CURRENT password:")
        contents = decrypt_file(".env.enc")
        # Write temp plaintext
        with open(".env.tmp", "w") as f:
            f.write(contents)
        print("\nEnter NEW password:")
        encrypt_file(".env.tmp", ".env.enc")
        Path(".env.tmp").unlink()
        print("Password changed.")
    else:
        print(f"Unknown command: {cmd}")
        print("Usage: python3 security.py [encrypt|decrypt|rekey]")


if __name__ == "__main__":
    main()
