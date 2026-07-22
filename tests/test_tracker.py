from drop_tracker.main import (
    Availability,
    detect_availability,
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
