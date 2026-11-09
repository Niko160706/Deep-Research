"""Manage provider credentials through a trusted operating-system keyring."""

from __future__ import annotations

import argparse
import getpass
import sys
from collections.abc import Callable

from gidr.infrastructure.security.secrets import ALIASES

_SERVICE = "gidr"
_TRUSTED_BACKENDS = {
    "keyring.backends.SecretService",
    "keyring.backends.kwallet",
    "keyring.backends.Windows",
    "keyring.backends.macOS",
}


def _trusted_keyring() -> object | None:
    try:
        import keyring

        backend = keyring.get_keyring()
    except Exception:
        return None
    if type(backend).__module__ not in _TRUSTED_BACKENDS:
        return None
    if getattr(backend, "priority", 0) <= 0:
        return None
    return keyring


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gidr-credential",
        description="Store provider API keys in the operating-system keyring.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="show configured aliases, never values")
    set_parser = commands.add_parser("set", help="securely prompt and store one key")
    set_parser.add_argument("alias", choices=sorted(ALIASES))
    delete_parser = commands.add_parser("delete", help="remove one stored key")
    delete_parser.add_argument("alias", choices=sorted(ALIASES))
    return parser


def run(
    argv: list[str] | None = None,
    *,
    keyring_loader: Callable[[], object | None] = _trusted_keyring,
    prompt: Callable[[str], str] = getpass.getpass,
    output: Callable[[str], None] = print,
) -> int:
    args = _parser().parse_args(argv)
    api = keyring_loader()
    if api is None:
        output("No trusted OS keyring is available. No credential was read or stored.")
        return 2

    if args.command == "status":
        for alias in sorted(ALIASES):
            try:
                configured = api.get_password(_SERVICE, alias) is not None  # type: ignore[attr-defined]
            except Exception:
                output("Keyring access failed; no credential value was displayed.")
                return 2
            output(f"{alias}: {'configured' if configured else 'not configured'}")
        return 0

    if args.command == "set":
        value = prompt(f"API key for {args.alias} (input hidden): ")
        confirmation = prompt("Confirm API key (input hidden): ")
        if not value or not value.strip() or value != confirmation:
            output("Credential was not stored: empty value or confirmation mismatch.")
            return 2
        try:
            api.set_password(_SERVICE, args.alias, value)  # type: ignore[attr-defined]
            stored = api.get_password(_SERVICE, args.alias)  # type: ignore[attr-defined]
        except Exception:
            output("Keyring write failed; credential value was not displayed.")
            return 2
        if stored != value:
            output("Keyring verification failed; credential value was not displayed.")
            return 2
        output(f"Stored and verified {args.alias} in the OS keyring.")
        return 0

    try:
        api.delete_password(_SERVICE, args.alias)  # type: ignore[attr-defined]
    except Exception:
        output("Keyring deletion failed; no credential value was displayed.")
        return 2
    output(f"Deleted {args.alias} from the OS keyring.")
    return 0


def main() -> int:
    try:
        return run()
    except (KeyboardInterrupt, EOFError):
        print("Credential operation cancelled; no value was displayed.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
