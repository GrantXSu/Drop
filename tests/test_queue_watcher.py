from drop_tracker.queue_watcher import QueueSignal, detect_queue_signal


def test_detects_queue_it_redirect() -> None:
    assert (
        detect_queue_signal(
            "https://pokemoncenter.queue-it.net/?c=store&e=drop",
            "Queue",
            "",
        )
        == QueueSignal.ACTIVE
    )


def test_detects_virtual_queue_copy() -> None:
    assert (
        detect_queue_signal(
            "https://www.pokemoncenter.com/",
            "Pokémon Center",
            "You are now in line. Your estimated wait time is 42 minutes.",
        )
        == QueueSignal.ACTIVE
    )


def test_distinguishes_antibot_block_from_queue() -> None:
    assert (
        detect_queue_signal(
            "https://www.pokemoncenter.com/",
            "Pardon Our Interruption",
            "Request unsuccessful. Incapsula incident ID: 123",
        )
        == QueueSignal.BLOCKED
    )


def test_normal_storefront_is_inactive() -> None:
    assert (
        detect_queue_signal(
            "https://www.pokemoncenter.com/",
            "Pokémon Center Official Site",
            "Shop Pokémon TCG, plush, clothing, and more.",
        )
        == QueueSignal.INACTIVE
    )
