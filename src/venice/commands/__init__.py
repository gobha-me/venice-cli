"""Single import point. Adding a subcommand = one import + one tuple entry."""
from . import balance, bg_remove, chat, embed, image, login, models, sfx, tts, upscale


def register_all(subparsers) -> None:
    login.register(subparsers)
    balance.register(subparsers)
    models.register(subparsers)
    sfx.register(subparsers)
    sfx.register_status(subparsers)
    chat.register(subparsers)
    tts.register(subparsers)
    image.register(subparsers)
    upscale.register(subparsers)
    bg_remove.register(subparsers)
    embed.register(subparsers)
