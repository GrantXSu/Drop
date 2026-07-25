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

## Run the card grade scanner

The optional CardLens web app accepts front and back photos, corrects the card
perspective, measures visible centering and wear signals, and estimates a
PSA-style 1–10 grade range.

```sh
source .venv/bin/activate
pip install -e '.[scanner]'
drop-card-grader
```

Open <http://127.0.0.1:8000>, then photograph one unsleeved card at a time on a
plain, contrasting background. Use diffuse light, keep all four corners
visible, and include both sides. The first run uses a clearly labeled,
low-confidence visual heuristic.

Choose **Use camera** for a live rear-camera preview with a card framing guide.
Capture the front and back, then choose **Analyze captured card** without saving
or selecting upload files. Browser camera access requires HTTPS or localhost;
the normal photo picker remains available when live camera access is blocked.

If an automatic centering guide follows text or a content bar, choose
**Enable line adjustment** in the inspection report. Drag the four green lines
directly on the normalized card image, then choose **Apply and recalculate**.
Adjustment mode is off by default. The updated guides, ratios, centering
subgrade, and overall estimate are recomputed from the explicit line positions.

For a single development command that opens the browser and reloads when source
files change:

```sh
drop-card-grader --reload --open
```

### Install CardLens as a Mac app

After completing the installation above, run this once:

```sh
scripts/install-cardlens-macos.sh
```

This installs a per-user background service that starts CardLens at login,
reloads after source updates, and creates `~/Applications/CardLens.app`. Open
that app like any other Mac application; no terminal needs to remain open.
CardLens still runs locally at <http://127.0.0.1:8000> and does not upload card
photos to a hosted service.

### Free and Pro plans

CardLens Free includes three unique card analyses per UTC day. Recalculating
manual guides or confirming a catalog match for the same captured card does not
consume another analysis. CardLens Pro is **$9.99/month** or **$59.99/year** and
includes unlimited analyses.

Payments use Stripe-hosted Checkout and Stripe's customer portal. Create one
monthly recurring price and one annual recurring price in Stripe, then set:

```sh
CARDLENS_COOKIE_SECRET="a-long-random-production-secret"
STRIPE_SECRET_KEY="sk_..."
STRIPE_WEBHOOK_SECRET="whsec_..."
STRIPE_PRO_MONTHLY_PRICE_ID="price_..."
STRIPE_PRO_ANNUAL_PRICE_ID="price_..."
```

Configure the webhook URL as
`https://your-host/api/billing/webhook` for checkout-session and subscription
created/updated/deleted events. For local testing, Stripe CLI can forward
events:

```sh
stripe listen --forward-to localhost:8000/api/billing/webhook
```

The included entitlement is tied to a signed browser/device cookie. Before a
commercial public launch, add user accounts and a hosted transactional database
so paid access works across devices and cannot be reset by clearing cookies.

### Sync the card identification catalog

CardLens can identify cards from every English series exposed by the
open-source [TCGdex](https://tcgdex.dev/) database:

```sh
drop-sync-card-catalog
```

The command stores card names, sets, numbers, reference URLs, visual
fingerprints, and clean-card condition baselines in
`data/grading/card_catalog.sqlite`. It does not retain copies of the reference
images outside the database; compact reference thumbnails are retained for
surface comparison. Re-run it to add newly released sets. The first complete
sync downloads and profiles many thousands of cards and can take a while.
Identification combines perceptual color/hash matching with local artwork
keypoints, but may still confuse parallel, reverse-holo, or similarly
illustrated printings. For confident matches, the clean reference layout
calibrates expected print placement and checks for localized surface anomalies
without moving the detected border guides.

When visual matching is not confident, the report provides a catalog search.
Enter a name, number, set, or combination such as `Pikachu 065`, then choose
**This is my card** to rerun the report with that exact reference.

### Train it with verified samples

Copy `examples/grading_manifest.csv` and add one row per graded card:

```csv
grading_company,overall_grade,bgs_corners,bgs_edges,front,back,source_url,usage_rights,certification_number
PSA,9,,,images/psa-front.jpg,images/psa-back.jpg,https://example.com/psa-cert,owner permission,12345678
BGS,9.5,9.5,9,images/bgs-front.jpg,images/bgs-back.jpg,https://example.com/bgs-cert,owner permission,87654321
```

`front` and `back` can be local paths relative to the CSV or direct public image
URLs. `source_url` records where the grade was verified, and `usage_rights`
records why the image can legally be used. `certification_number` prevents
unverifiable labels. PSA rows train the overall PSA target. PSA does not publish
numeric corner or edge subgrades, so those fields must remain blank for PSA.
BGS rows can train separate corner and edge targets from slab subgrades. Then
run:

```sh
drop-train-grader path/to/grading_manifest.csv
drop-card-grader
```

The trainer fits separate `psa_overall`, `bgs_corners`, and `bgs_edges` targets.
Each target independently requires at least 100 labels from 80 distinct cards
across four grade bands. It rejects blurry, overexposed, underexposed, and
glare-obscured images, holds out entire source cards, reports target-specific
mean absolute error, and saves `models/card_grader.joblib`. BGS category models
are used only when holdout MAE is 1.0 or better. Capture-quality signals affect
confidence but are not model inputs or physical-damage penalties. A
production-quality model still needs hundreds or thousands of diverse,
correctly labeled front-and-back examples.

Publicly viewable PSA grades and images are not automatically licensed for
bulk scraping or model training. The project therefore imports an explicit
provenance manifest instead of scraping PSA. Check each source's terms and get
permission where needed.

CardLens is not affiliated with PSA and cannot inspect damage hidden by glare,
sleeves, holders, or image resolution. Its result is an estimate, not a
certification or guarantee of the grade a grading company will assign.

Centering calibration follows the published [PSA grading
standards](https://www.psacard.com/gradingstandards) and [Beckett grading
scale](https://www.beckett.com/grading/scale). PSA permits approximately 55/45
front and 75/25 reverse centering for Gem Mint 10. Beckett requires 50/50 on
the front for Pristine 10 and publishes separate front/back thresholds for
lower grades. The app reports a decimal PSA-style estimate and a separate
Beckett-style centering reference; neither is an official grade. The decimal
estimate is intentionally conservative: a back image is required for 10.0,
both sides affect the score, and merely landing on PSA's maximum tolerance
does not guarantee 10.0. Repeated confirmed whitening regions increase edge
and corner penalties even when each chip is small. Centering contributes 10%
of the heuristic weighting and does not impose the severe physical-damage cap.

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
drop-card-grader                 Open the local card grading web app
drop-train-grader MANIFEST.csv   Train on verified sample images and grades
```
