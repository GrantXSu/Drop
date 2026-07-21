# Drop Tracker

A notification-only Pokémon Center stock tracker. It checks at a conservative
interval, alerts through Discord when availability changes, and leaves checkout
entirely manual. No browser page needs to remain open when it runs in the cloud.

## What it does

- Searches the official Pokémon Center TCG category for 30th Celebration Elite
  Trainer Boxes, Booster Bundles, and 3-Pack products.
- Supports exact product URLs as soon as Pokémon Center publishes them.
- Recognizes structured product availability and enabled Add to Cart/Preorder
  buttons.
- Sends an alert with a clickable official product link only when a product
  changes to available.
- Treats blocks, rate limits, and ambiguous pages as unknown instead of sending
  false availability alerts.

The tracker does not bypass queues or CAPTCHAs and does not automate checkout.
Keep the default five-minute interval or make it longer. Repeated rapid requests
can trigger Pokémon Center's protections.

## Configure it

1. Copy the environment template:

   ```sh
   cp .env.example .env
   ```

2. Create a Discord webhook under **Server Settings → Integrations → Webhooks**
   and put its URL in `DISCORD_WEBHOOK_URL`.

3. When an official product page is known, add it to `TARGET_URLS`. Multiple
   URLs can be comma-separated. Exact URLs are more reliable than discovery:

   ```dotenv
   TARGET_URLS=https://www.pokemoncenter.com/product/...
   ```

Secrets belong only in `.env` or your cloud host's secret settings. Never commit
the `.env` file.

## Test locally

Python 3.9 or newer:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
set -a; source .env; set +a
drop-tracker --test-notification
drop-tracker --once
pytest
```

To run continuously with Docker:

```sh
docker compose up --build -d
docker compose logs -f tracker
```

## Run free with GitHub Actions

The included workflow checks every five minutes without requiring your computer
to stay on:

1. Push this project to a public GitHub repository.
2. Open **Settings → Secrets and variables → Actions → Secrets**.
3. Create a repository secret named `DISCORD_WEBHOOK_URL`.
4. Open **Actions → Track Pokemon Center drops → Run workflow**.
5. Enable `Send a test Discord alert` and run it.

Official product URLs can be added later under **Settings → Secrets and
variables → Actions → Variables** as a variable named `TARGET_URLS`. Separate
multiple URLs with commas.

GitHub schedules can start late during busy periods, so a five-minute schedule
is not a guarantee that every short-lived restock will be caught. GitHub may
disable scheduled workflows on public repositories after 60 days without
repository activity; re-enable the workflow from the Actions page if that
happens.

## Run continuously on Render

`render.yaml` defines a Render background worker:

1. Put this project in a private GitHub repository.
2. In Render, choose **New → Blueprint** and connect the repository.
3. Enter the requested Discord webhook secret.
4. Deploy, then check the worker logs.

The included worker uses Render's paid Starter plan because continuously running
free workers are not generally available. The Docker image can instead run on
Railway, Fly.io, a VPS, a NAS, or any always-on Docker host using the same
environment variables.

Cloud containers may lose `data/state.json` during a redeploy unless persistent
storage is attached. The only consequence is one repeat alert if the product is
already available when the replacement starts.

## Commands

```text
drop-tracker                     Run continuously
drop-tracker --once              Check once and exit
drop-tracker --test-notification Test Discord, then exit
```
