from drop_tracker.main import (
    Availability,
    Config,
    ProductListing,
    Tracker,
    detect_availability,
    discover_product_listings,
    discover_product_urls,
)


def test_detects_json_ld_in_stock() -> None:
    page = """
    <script type="application/ld+json">
      {"@type": "Product", "offers": {"availability": "https://schema.org/InStock"}}
    </script>
    """
    assert detect_availability(page) == Availability.AVAILABLE


def test_detects_enabled_add_to_cart() -> None:
    assert (
        detect_availability("<button>Add to Cart</button>")
        == Availability.AVAILABLE
    )


def test_disabled_add_to_cart_is_not_available() -> None:
    page = "<button disabled>Add to Cart</button><p>Sold out</p>"
    assert detect_availability(page) == Availability.UNAVAILABLE


def test_unknown_page_does_not_create_false_positive() -> None:
    assert detect_availability("<h1>Please wait</h1>") == Availability.UNKNOWN


def test_discovers_matching_official_product_links() -> None:
    page = """
    <a href="/product/123/pokemon-30th-celebration-elite-trainer-box">
      Pokémon TCG: 30th Celebration Elite Trainer Box
    </a>
    <a href="/product/456/pokemon-30th-celebration-booster-bundle">
      30th Celebration Booster Bundle
    </a>
    <a href="/product/457/pokemon-30th-celebration-3-pack-blister">
      30th Celebration 3-Pack Blister
    </a>
    <a href="/product/458/pokemon-30th-celebration-poster">
      30th Celebration Poster
    </a>
    <a href="https://example.com/product/789/30th-celebration-elite-trainer-box">
      30th Celebration Elite Trainer Box
    </a>
    """
    assert discover_product_urls(
        page,
        "https://www.pokemoncenter.com/category/trading-card-game",
        ("30th celebration",),
        ("elite trainer box", "booster bundle", "3 pack"),
    ) == {
        "https://www.pokemoncenter.com/product/123/"
        "pokemon-30th-celebration-elite-trainer-box",
        "https://www.pokemoncenter.com/product/456/"
        "pokemon-30th-celebration-booster-bundle",
        "https://www.pokemoncenter.com/product/457/"
        "pokemon-30th-celebration-3-pack-blister",
    }


def test_discovers_additional_30th_collection_products() -> None:
    product_names = (
        "Sylveon ex Box",
        "Greninja ex Box",
        "Poster Collection",
        "Binder Collection",
        "Mew Figure Collection",
        "Mewtwo Figure Collection",
        "Ditto Premium Collection",
    )
    page = "".join(
        f'<a href="/product/{index}/30th-celebration-{name.casefold().replace(" ", "-")}">'
        f"30th Celebration {name}</a>"
        for index, name in enumerate(product_names, start=1)
    )

    matches = discover_product_urls(
        page,
        "https://www.pokemoncenter.com/category/trading-card-game",
        ("30th celebration",),
        (
            "sylveon ex",
            "greninja ex",
            "poster collection",
            "binder collection",
            "mew figure collection",
            "mewtwo figure collection",
            "ditto premium collection",
        ),
    )

    assert len(matches) == len(product_names)


def test_listing_parser_detects_stock_and_tcg_priority() -> None:
    page = """
    <a href="/product/100/tcg-box">
      <img alt="Pokémon TCG: Example Elite Trainer Box">SOLD OUT
    </a>
    <a href="/product/100/tcg-box">Pokémon TCG: Example Elite Trainer Box $59.99</a>
    <a href="/product/200/pikachu-plush">Pikachu Plush $24.99</a>
    """
    listings = discover_product_listings(
        page, "https://www.pokemoncenter.com/category/new-releases"
    )

    tcg = listings["https://www.pokemoncenter.com/product/100/tcg-box"]
    plush = listings["https://www.pokemoncenter.com/product/200/pikachu-plush"]
    assert tcg.availability == Availability.UNAVAILABLE
    assert tcg.is_tcg is True
    assert plush.availability == Availability.AVAILABLE
    assert plush.is_tcg is False


def test_tracker_baselines_then_alerts_for_new_and_restocked_products(
    tmp_path,
) -> None:
    config = Config(
        target_urls=(),
        discovery_urls=(),
        match_terms=(),
        product_terms=(),
        track_all_products=True,
        interval_seconds=300,
        jitter_seconds=30,
        request_timeout=20,
        state_file=tmp_path / "state.json",
        discord_webhook_url=None,
        smtp_host=None,
        smtp_port=587,
        smtp_username=None,
        smtp_password=None,
        email_from=None,
        email_to=(),
    )
    tracker = Tracker(config)
    alerts = []
    tracker.notify = lambda *args, **kwargs: alerts.append((args, kwargs))
    existing = ProductListing(
        "https://www.pokemoncenter.com/product/1/existing",
        "Existing Plush",
        Availability.UNAVAILABLE,
        False,
    )
    tracker._discover_listings = lambda: {existing.url: existing}
    tracker.check_once()
    assert alerts == []

    restocked = ProductListing(
        existing.url, existing.title, Availability.AVAILABLE, False
    )
    new_tcg = ProductListing(
        "https://www.pokemoncenter.com/product/2/new-tcg",
        "Pokémon TCG: New Booster Bundle",
        Availability.AVAILABLE,
        True,
    )
    tracker._discover_listings = lambda: {
        restocked.url: restocked,
        new_tcg.url: new_tcg,
    }
    tracker.check_once()
    tracker.close()

    assert len(alerts) == 2
    assert any(kwargs["is_tcg"] for _, kwargs in alerts)
    assert any("RESTOCK" in args[0] for args, _ in alerts)
    assert any("NEW DROP" in args[0] for args, _ in alerts)
