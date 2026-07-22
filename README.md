# Drop Tracker

A notification-only Pokémon Center stock tracker. It checks at a conservative
interval, alerts through Discord when availability changes, and leaves checkout
entirely manual. No browser page needs to remain open when it runs in the cloud.

## What it does

- Runs a visible local Chrome session that detects Pokémon Center's virtual
  waiting room and sends an urgent Discord early-warning alert.
- Preserves the live queue page without refreshing after a queue is detected.
- Distinguishes the virtual queue from Error 17 and anti-bot block pages.
- Watches Pokémon Center's New Releases, TCG, Plush, and Figures & Pins category
  pages for newly listed products and restocks.
- Gives TCG products prominent priority alerts while still notifying for
  plushies, figures, pins, and other merchandise shown in New Releases.
- Supports exact product URLs as soon as Pokémon Center publishes them.
- Sends clickable alerts for newly available listings and transitions from
  sold out to available.
- Saves the first broad scan as a baseline without flooding Discord with every
  product that was already listed.
- Treats blocks, rate limits, and ambiguous pages as unknown instead of sending
  false availability alerts.

The tracker does not bypass queues or CAPTCHAs and does not automate checkout.
Keep the default five-minute interval or make it longer. Repeated rapid requests
can trigger Pokémon Center's protections.

Category pages do not expose the entire historical Pokémon Center catalog at
once. The tracker catches products visible on its monitored pages; an older
product outside those pages needs to be added to `TARGET_URLS` for direct
restock monitoring. Pokémon Center also uses anti-bot protection and may block
cloud checks even at a conservative rate. The tracker does not bypass that
protection; blocked checks are logged and never produce false stock alerts.

Pokémon Center officially states that a virtual queue does not necessarily mean
a product is launching. Treat the queue notification as an urgent early warning,
not confirmation of inventory.

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

## Run the queue detector

The local queue detector is the recommended setup because Pokémon Center blocks
GitHub's cloud IP addresses and GitHub schedules can run hours late.

```sh
cd ~/Drop
source .venv/bin/activate
pip install -e '.[dev]'
set -a; source .env; set +a
drop-queue-watcher --test-notification
drop-queue-watcher
```

A dedicated Chrome window opens and reloads Pokémon Center approximately every
90–105 seconds. Keep Terminal, Chrome, and the computer running. When the queue
appears, the watcher sends Discord a clickable warning, brings Chrome forward,
and stops reloading so the live queue session remains intact. It never bypasses
the queue or automates checkout.

To run continuously with Docker:

```sh
docker compose up --build -d
docker compose logs -f tracker
```

## Manual GitHub Actions fallback

The GitHub workflow is manual only. Scheduled checks were disabled because
GitHub started them hours late and Pokémon Center timed out every cloud request.

1. Push this project to a public GitHub repository.
2. Open **Settings → Secrets and variables → Actions → Secrets**.
3. Create a repository secret named `DISCORD_WEBHOOK_URL`.
4. Open **Actions → Track Pokemon Center drops → Run workflow**.
5. Enable `Send a test Discord alert` and run it.

Official product URLs can be added later under **Settings → Secrets and
variables → Actions → Variables** as a variable named `TARGET_URLS`. Separate
multiple URLs with commas.

Do not rely on GitHub Actions for live drop detection.

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
drop-queue-watcher               Watch the virtual queue in local Chrome
```
