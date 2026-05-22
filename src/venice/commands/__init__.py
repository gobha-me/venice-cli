"""Single import point. Adding a subcommand = one import + one tuple entry."""
from . import chat, embed, image, login, sfx, tts


def register_all(subparsers) -> None:
    login.register(subparsers)
    sfx.register(subparsers)
    sfx.register_status(subparsers)
    chat.register(subparsers)
    tts.register(subparsers)
    image.register(subparsers)
    embed.register(subparsers)
