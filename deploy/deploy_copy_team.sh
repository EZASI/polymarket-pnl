#!/bin/bash
# =============================================================
# DEPLOY COPY TRADING TEAM — Replace losing bots with copy executor
# =============================================================
#
# What this does:
#   1. STOPS losing services: tournament (30 bots, all losing) + cross-platform arb (no edge)
#   2. DEPLOYS copy executor as a systemd service (the actual money-maker)
#   3. UPDATES dashboard with new TRADERS tab (copy signals view)
#   4. KEEPS: copy_signal_bot cron (every 10 min), trader_tracker cron (every 10 min)
#   5. SETS UP agent team: signal scanner → copy executor → dashboard
#
# Architecture (Agent Team):
#   ┌─────────────────┐    ┌──────────────────┐    ┌──────────────┐
#   │ copy_signal_bot  │───▶│  copy_executor    │───▶│  dashboard   │
#   │ (cron: 10 min)   │    │  (systemd: loop)  │    │  (port 8080) │
#   │ Scans top traders │    │  Kelly sizing     │    │  Copy signals│
#   │ Finds convergence │    │  Paper/Live mode  │    │  Trader view │
#   └─────────────────┘    └──────────────────┘    └──────────────┘
#          │                        │
#          ▼                        ▼
#   ┌─────────────────┐    ┌──────────────────┐
#   │ trader_tracker   │    │ logs/copy_trades  │
#   │ (cron: 10 min)   │    │ (trade history)   │
#   └─────────────────┘    └──────────────────┘
#
# Usage:
#   ssh -i ~/Downloads/polymarket-key.pem ubuntu@43.207.206.223
#   cd polymarket-pnl
#   bash deploy/deploy_copy_team.sh
#
#   # For live trading:
#   bash deploy/deploy_copy_team.sh --live
#
# =============================================================

set -eo pipefail

LIVE_MODE="${1:-paper}"
WORK_DIR="/home/ubuntu/polymarket-pnl"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  MTY COPY TRADING TEAM — DEPLOYMENT                        ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Stops: tournament (30 bots), cross-platform arb           ║"
echo "║  Starts: copy executor (follows top traders)               ║"
echo "║  Keeps: signal scanner, tracker, dashboard                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── 1. Stop losing services ──
echo "[1/6] Stopping losing services..."
echo "  Stopping polymarket-tournament (30 bots, all losing ~$96k)..."
sudo systemctl stop polymarket-tournament 2>/dev/null || true
sudo systemctl disable polymarket-tournament 2>/dev/null || true
echo "  Stopping crossplatform-arb (no real edge found)..."
sudo systemctl stop crossplatform-arb 2>/dev/null || true
sudo systemctl disable crossplatform-arb 2>/dev/null || true
echo "  Done. Freed CPU/RAM for copy executor."

# ── 2. Install Python deps ──
echo ""
echo "[2/6] Checking Python dependencies..."
cd "$WORK_DIR"
source venv/bin/activate

pip install requests pandas numpy py-clob-client python-dotenv 2>/dev/null | tail -1
echo "  Dependencies OK."

# ── 3. Install copy executor service ──
echo ""
echo "[3/6] Installing copy-executor systemd service..."

EXEC_ARGS=""
if [ "$LIVE_MODE" = "--live" ]; then
    EXEC_ARGS="--live --bankroll 50000"
    echo "  MODE: LIVE TRADING (real USDC)"
else
    EXEC_ARGS="--bankroll 50000"
    echo "  MODE: PAPER TRADING (safe, no real money)"
fi

sudo tee /etc/systemd/system/copy-executor.service > /dev/null <<SERVICE
[Unit]
Description=MTY Copy Trading Executor
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=$WORK_DIR
ExecStart=$WORK_DIR/venv/bin/python3 copy_executor.py $EXEC_ARGS
Restart=always
RestartSec=30
Nice=-5
Environment=PYTHONUNBUFFERED=1
Environment=ENC_PASSWORD=PolyBot2026Secure
StandardOutput=append:$WORK_DIR/logs/copy_executor.log
StandardError=append:$WORK_DIR/logs/copy_executor.err
LimitNOFILE=65535
MemoryMax=1G

[Install]
WantedBy=multi-user.target
SERVICE

sudo systemctl daemon-reload
sudo systemctl enable copy-executor
echo "  Service installed."

# ── 4. Restart dashboard with updated code ──
echo ""
echo "[4/6] Restarting dashboard..."
sudo systemctl restart polymarket-dashboard
echo "  Dashboard restarted at http://$(curl -s ifconfig.me 2>/dev/null || echo '43.207.206.223'):8080"

# ── 5. Verify cron jobs ──
echo ""
echo "[5/6] Verifying cron jobs..."
CRON_OK=true

if crontab -l 2>/dev/null | grep -q "copy_signal_bot"; then
    echo "  copy_signal_bot: every 10 min (OK)"
else
    echo "  copy_signal_bot: MISSING — adding..."
    (crontab -l 2>/dev/null; echo "*/10 * * * * cd $WORK_DIR && $WORK_DIR/venv/bin/python3 copy_signal_bot.py >> logs/copy_cron.log 2>&1") | crontab -
    CRON_OK=false
fi

if crontab -l 2>/dev/null | grep -q "trader_tracker"; then
    echo "  trader_tracker:  every 10 min (OK)"
else
    echo "  trader_tracker: MISSING — adding..."
    (crontab -l 2>/dev/null; echo "*/10 * * * * cd $WORK_DIR && $WORK_DIR/venv/bin/python3 trader_tracker.py >> logs/tracker_cron.log 2>&1") | crontab -
fi

# ── 6. Start copy executor ──
echo ""
echo "[6/6] Starting copy executor..."
sudo systemctl start copy-executor
sleep 2
STATUS=$(sudo systemctl is-active copy-executor 2>/dev/null || echo "failed")

if [ "$STATUS" = "active" ]; then
    echo "  copy-executor: RUNNING"
else
    echo "  copy-executor: $STATUS"
    echo "  Check: sudo journalctl -u copy-executor -n 20"
fi

# ── Summary ──
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  DEPLOYMENT COMPLETE — MTY COPY TRADING TEAM               ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║                                                              ║"
echo "║  RUNNING SERVICES:                                           ║"
echo "║    copy-executor     — follows top traders (paper/live)      ║"
echo "║    polymarket-dashboard — web UI at :8080                    ║"
echo "║                                                              ║"
echo "║  CRON JOBS (every 10 min):                                   ║"
echo "║    copy_signal_bot   — scans top traders for convergence     ║"
echo "║    trader_tracker    — tracks top 5 trader positions         ║"
echo "║                                                              ║"
echo "║  STOPPED (waste of compute):                                 ║"
echo "║    polymarket-tournament — 30 crypto bots (all losing)       ║"
echo "║    crossplatform-arb     — no real edge found                ║"
echo "║                                                              ║"
echo "║  COMMANDS:                                                   ║"
echo "║    tail -f logs/copy_executor.log   # watch trades           ║"
echo "║    tail -f logs/copy_cron.log       # watch signal scanner   ║"
echo "║    sudo systemctl status copy-executor                       ║"
echo "║    sudo systemctl restart copy-executor                      ║"
echo "║                                                              ║"
echo "║  TO SWITCH TO LIVE:                                          ║"
echo "║    Edit /etc/systemd/system/copy-executor.service            ║"
echo "║    Add --live to ExecStart line                              ║"
echo "║    sudo systemctl daemon-reload && sudo systemctl restart    ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Show current service status
echo "SERVICE STATUS:"
echo "  copy-executor:         $(sudo systemctl is-active copy-executor 2>/dev/null || echo 'inactive')"
echo "  polymarket-dashboard:  $(sudo systemctl is-active polymarket-dashboard 2>/dev/null || echo 'inactive')"
echo "  polymarket-tournament: $(sudo systemctl is-active polymarket-tournament 2>/dev/null || echo 'inactive') (should be inactive)"
echo "  crossplatform-arb:     $(sudo systemctl is-active crossplatform-arb 2>/dev/null || echo 'inactive') (should be inactive)"
echo ""
