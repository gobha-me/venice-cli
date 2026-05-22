"""Single import point. Adding a subcommand = one import + one tuple entry."""
from . import balance, chat, embed, image, login, models, sfx, tts


def register_all(subparsers) -> None:
    login.register(subparsers)
    balance.register(subparsers)
    models.register(subparsers)
    sfx.register(subparsers)
    sfx.register_status(subparsers)
    chat.register(subparsers)
    tts.register(subparsers)
    image.register(subparsers)
    embed.register(subparsers)
