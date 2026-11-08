"""Side-effect-free composition boundary for the current contracts batch."""

from gidr.config import AppSettings, load_settings


def configure() -> AppSettings:
    return load_settings()


def main() -> int:
    configure()
    print("Konfigurasi valid. Mesin riset belum diaktifkan pada tahap ini.")
    return 0
