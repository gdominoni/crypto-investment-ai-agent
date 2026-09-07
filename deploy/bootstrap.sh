#!/usr/bin/env bash
#
# One-shot host setup: system packages, the code, a virtualenv with the lean
# dependency set, and a systemd unit with this machine's real user and paths
# filled in.
#
# Run it on the HOST, as your normal login user (`ubuntu` on Oracle's images --
# not root, and not with sudo):
#
#     curl -fsSL https://raw.githubusercontent.com/gdominoni/crypto-investment-ai-agent/main/deploy/bootstrap.sh | bash
#
# or, if you prefer to read it first (reasonable, since it uses sudo):
#
#     curl -fsSLO https://raw.githubusercontent.com/gdominoni/crypto-investment-ai-agent/main/deploy/bootstrap.sh
#     less bootstrap.sh && bash bootstrap.sh
#
# It deliberately stops short of starting anything. Two things still have to
# come from your own machine first -- the secrets and the accumulated history --
# and a daemon started before those arrive is a daemon that looks healthy while
# doing nothing useful. See deploy/README.md steps 3 to 6.
#
# Safe to re-run: it skips what is already done rather than starting over.
set -euo pipefail

REPO_URL="https://github.com/gdominoni/crypto-investment-ai-agent.git"
TARGET_DIR="${TARGET_DIR:-$HOME/crypto-agent}"
SERVICE_NAME="crypto-agent"

# The lean set, and not requirements.txt. Nothing the daemon runs imports
# freqtrade or scipy -- they belong to the local hyperopt cross-check and to
# the offline analyses in forecast/ -- and they cost several hundred MB here
# for nothing. See deploy/README.md.
HOST_PACKAGES=(pandas numpy pyarrow requests anthropic python-dotenv ccxt)

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

[ "$(id -u)" -ne 0 ] || die "run this as your normal login user (ubuntu), not as root."
command -v sudo >/dev/null || die "sudo not found."

say "1/5  System packages"
# Oracle's images run unattended-upgrades on first boot, which holds the apt
# lock for a few minutes. Waiting beats failing with a confusing lock error.
if command -v apt-get >/dev/null; then
    for _ in $(seq 1 30); do
        sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 || break
        echo "    waiting for another apt process to finish..."
        sleep 10
    done
    sudo apt-get update -qq
    sudo apt-get install -y -qq python3-venv python3-pip git
else
    die "this script expects a Debian/Ubuntu host (apt-get not found)."
fi
echo "    python3: $(python3 --version), arch: $(uname -m)"

say "2/5  Code"
if [ -d "$TARGET_DIR/.git" ]; then
    echo "    already cloned, pulling instead"
    git -C "$TARGET_DIR" pull --ff-only
else
    git clone --quiet "$REPO_URL" "$TARGET_DIR"
    echo "    cloned into $TARGET_DIR"
fi

say "3/5  Virtualenv and dependencies"
if [ ! -d "$TARGET_DIR/.venv" ]; then
    python3 -m venv "$TARGET_DIR/.venv"
fi
"$TARGET_DIR/.venv/bin/pip" install --quiet --upgrade pip
# On ARM (Oracle's A1 shapes) this pulls aarch64 wheels; no compiler needed.
"$TARGET_DIR/.venv/bin/pip" install --quiet "${HOST_PACKAGES[@]}"
echo "    installed: ${HOST_PACKAGES[*]}"

say "4/5  Verify the daemon imports"
# Catches a missing dependency now, on a prompt you are watching, rather than
# in a journal you are not.
( cd "$TARGET_DIR" && "$TARGET_DIR/.venv/bin/python3" -c "import scheduler.live_daemon" ) \
    && echo "    the daemon and everything it imports resolve" \
    || die "the daemon failed to import -- the package list above is missing something."

say "5/5  systemd unit"
# Generated rather than hand-edited: the stock unit carries placeholder paths,
# and a wrong WorkingDirectory fails in a way that reads like a code bug.
sed -e "s|^User=.*|User=$(id -un)|" \
    -e "s|^WorkingDirectory=.*|WorkingDirectory=${TARGET_DIR}|" \
    -e "s|^ExecStart=.*|ExecStart=${TARGET_DIR}/.venv/bin/python3 -m scheduler.live_daemon|" \
    -e "s|^ReadWritePaths=.*|ReadWritePaths=${TARGET_DIR}|" \
    "$TARGET_DIR/deploy/${SERVICE_NAME}.service" | sudo tee "/etc/systemd/system/${SERVICE_NAME}.service" >/dev/null
sudo systemctl daemon-reload
echo "    installed /etc/systemd/system/${SERVICE_NAME}.service for user $(id -un)"

cat <<EOF

Host is ready. NOT started yet, on purpose -- two things still have to arrive
from your own machine, and starting without them gives you a daemon that looks
perfectly healthy while doing nothing useful.

  1. Secrets.  Create ${TARGET_DIR}/.env with ANTHROPIC_API_KEY,
     TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID (and HEARTBEAT_URL if you use a
     dead-man's switch), then: chmod 600 ${TARGET_DIR}/.env

  2. History.  Copy the runtime state across -- deploy/README.md step 4 has
     the exact scp commands. Skip only if this host should start from nothing.

Then, on this host:

     cd ${TARGET_DIR}
     .venv/bin/python3 -m data_ingestion.market_data.binance_fetcher
     .venv/bin/python3 -m deploy.preflight

Every line must say PASS. When it does:

     sudo systemctl enable --now ${SERVICE_NAME}
     journalctl -u ${SERVICE_NAME} -f
EOF
