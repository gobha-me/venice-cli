"""Single import point. Adding a subcommand = one import + one tuple entry."""
from . import balance, bg_remove, chat, code, config, contact_sheet, embed, image, index, login, master, mcp_serve, models, music, search, sfx, tts, upscale, video


def register_all(subparsers) -> None:
    login.register(subparsers)
    balance.register(subparsers)
    models.register(subparsers)
    sfx.register(subparsers)
    sfx.register_status(subparsers)
    music.register(subparsers)
    music.register_status(subparsers)
    video.register(subparsers)
    video.register_status(subparsers)
    chat.register(subparsers)
    tts.register(subparsers)
    image.register(subparsers)
    upscale.register(subparsers)
    bg_remove.register(subparsers)
    embed.register(subparsers)
    index.register(subparsers)
    search.register(subparsers)
    code.register(subparsers)
    master.register(subparsers)
    contact_sheet.register(subparsers)
    mcp_serve.register(subparsers)
    config.register(subparsers)
