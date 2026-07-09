#!/bin/bash
# =============================================================
# MAX SPEED DEPLOYMENT — AWS c5.large in ap-northeast-1 (Tokyo)
# =============================================================
#
# Instance:  c5.large (2 vCPU, 4GB RAM, up to 10 Gbps network)
# Region:    ap-northeast-1 (Tokyo) — 2-5ms to Binance
# AMI:       Ubuntu 24.04 LTS
# Storage:   30GB gp3 (3000 IOPS)
# Cost:      ~$62/month
#
# This script:
#   1. Installs Python 3.11+ with all dependencies
#   2. Tunes the Linux kernel for low-latency networking
#   3. Sets up the bot as a systemd service (auto-restart)
#   4. Configures log rotation
#   5. Runs latency test to Binance
#
# USAGE:
#   1. Launch c5.large in ap-northeast-1 (Ubuntu 24.04)
#   2. SCP your code:
#        scp -i key.pem -r polymarket-pnl ubuntu@IP:/home/ubuntu/
#   3. SSH in:
#        ssh -i key.pem ubuntu@IP
#   4. Run:
#        cd polymarket-pnl && bash deploy/deploy_max_speed.sh
#   5. Start:
#        sudo systemctl start polymarket-bot
# =============================================================

set -euo pipefail

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  MAX SPEED DEPLOYMENT — Polymarket Trading Bot              ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# ── 1. System updates ──
echo ""
echo "[1/7] Updating system..."
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get upgrade -y -qq

# ── 2. Install Python + build tools ──
echo ""
echo "[2/7] Installing Python and build tools..."
sudo apt-get install -y -qq \
    python3 python3-pip python3-venv python3-dev \
    git curl build-essential \
    htop iotop net-tools

# ── 3. Kernel tuning for low-latency networking ──
echo ""
echo "[3/7] Tuning kernel for low-latency networking..."
sudo tee /etc/sysctl.d/99-low-latency.conf > /dev/null <<'SYSCTL'
# TCP tuning for low-latency trading
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
net.core.rmem_default = 1048576
net.core.wmem_default = 1048576
net.ipv4.tcp_rmem = 4096 1048576 16777216
net.ipv4.tcp_wmem = 4096 1048576 16777216
net.ipv4.tcp_timestamps = 1
net.ipv4.tcp_sack = 1
net.ipv4.tcp_window_scaling = 1
net.ipv4.tcp_fastopen = 3
net.ipv4.tcp_nodelay = 1
net.ipv4.tcp_low_latency = 1
net.core.netdev_max_backlog = 5000
net.core.somaxconn = 65535

# Reduce swap usage (keep everything in RAM)
vm.swappiness = 10
vm.dirty_ratio = 10
vm.dirty_background_ratio = 5
SYSCTL
sudo sysctl --system > /dev/null 2>&1

# ── 4. Set up Python environment ──
echo ""
echo "[4/7] Setting up Python environment..."
cd /home/ubuntu/polymarket-pnl

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip > /dev/null 2>&1

echo "  Installing dependencies..."
pip install \
    requests aiohttp websockets \
    numpy pandas pyarrow \
    scikit-learn lightgbm xgboost joblib \
    py-clob-client python-dotenv \
    > /dev/null 2>&1

echo "  Dependencies installed."

# ── 5. Create directories ──
mkdir -p logs models

# ── 6. Install systemd service ──
echo ""
echo "[5/7] Installing systemd service..."

sudo tee /etc/systemd/system/polymarket-bot.service > /dev/null <<SERVICE
[Unit]
Description=Polymarket Trading Bot — Max Speed
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/polymarket-pnl
ExecStart=/home/ubuntu/polymarket-pnl/venv/bin/python3 polymarket_trader.py
Restart=always
RestartSec=5

# Performance
Nice=-10
CPUSchedulingPolicy=fifo
CPUSchedulingPriority=50

# Environment
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONDONTWRITEBYTECODE=1

# Logging
StandardOutput=append:/home/ubuntu/polymarket-pnl/logs/bot.log
StandardError=append:/home/ubuntu/polymarket-pnl/logs/bot.err

# Resource limits
LimitNOFILE=65535
MemoryMax=3G

[Install]
WantedBy=multi-user.target
SERVICE

# Also install streaming-only service (without Polymarket trading)
sudo tee /etc/systemd/system/streaming-alpha.service > /dev/null <<SERVICE2
[Unit]
Description=Streaming Alpha Engine — Signals Only
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/polymarket-pnl
ExecStart=/home/ubuntu/polymarket-pnl/venv/bin/python3 streaming_predictor.py --self-train
Restart=always
RestartSec=5
Nice=-10
Environment=PYTHONUNBUFFERED=1
StandardOutput=append:/home/ubuntu/polymarket-pnl/logs/streaming.log
StandardError=append:/home/ubuntu/polymarket-pnl/logs/streaming.err
LimitNOFILE=65535
MemoryMax=3G

[Install]
WantedBy=multi-user.target
SERVICE2

sudo systemctl daemon-reload
sudo systemctl enable polymarket-bot.service

# ── 7. SECURITY: Firewall + fail2ban ──
echo ""
echo "[6/8] Hardening security..."

# UFW Firewall
echo "  Configuring UFW firewall..."
sudo apt-get install -y -qq ufw
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp comment 'SSH'
sudo ufw allow 8080/tcp comment 'Dashboard'
sudo ufw --force enable
echo "  UFW enabled: SSH(22) + Dashboard(8080) only"

# fail2ban — block brute force SSH
echo "  Installing fail2ban..."
sudo apt-get install -y -qq fail2ban

sudo tee /etc/fail2ban/jail.local > /dev/null <<'F2B'
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 3

[sshd]
enabled = true
port = ssh
filter = sshd
logpath = /var/log/auth.log
maxretry = 3
bantime = 3600
F2B

sudo systemctl enable fail2ban
sudo systemctl restart fail2ban
echo "  fail2ban enabled: 3 failed SSH attempts = 1 hour ban"

# File permissions
chmod 700 /home/ubuntu/polymarket-pnl
chmod 600 /home/ubuntu/polymarket-pnl/.env 2>/dev/null || true
chmod 600 /home/ubuntu/polymarket-pnl/.env.enc 2>/dev/null || true
echo "  File permissions hardened"

# ── 8. Log rotation ──
echo ""
echo "[7/8] Setting up log rotation..."
sudo tee /etc/logrotate.d/polymarket > /dev/null <<LOGROTATE
/home/ubuntu/polymarket-pnl/logs/*.log
/home/ubuntu/polymarket-pnl/logs/*.err {
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
}
LOGROTATE

# ── 9. Latency test ──
echo ""
echo "[8/8] Testing latency to Binance..."
echo ""
echo "  Binance API (api.binance.com):"
for i in 1 2 3 4 5; do
    LATENCY=$(curl -o /dev/null -s -w '%{time_total}' https://api.binance.com/api/v3/ping)
    MS=$(echo "$LATENCY * 1000" | bc 2>/dev/null || echo "$LATENCY")
    echo "    Ping $i: ${MS}ms"
done

echo ""
echo "  Binance WebSocket (stream.binance.com):"
for i in 1 2 3; do
    LATENCY=$(curl -o /dev/null -s -w '%{time_connect}' https://stream.binance.com:9443)
    MS=$(echo "$LATENCY * 1000" | bc 2>/dev/null || echo "$LATENCY")
    echo "    Connect $i: ${MS}ms"
done

echo ""
echo "  Polymarket API (clob.polymarket.com):"
for i in 1 2 3; do
    LATENCY=$(curl -o /dev/null -s -w '%{time_total}' https://clob.polymarket.com/time)
    MS=$(echo "$LATENCY * 1000" | bc 2>/dev/null || echo "$LATENCY")
    echo "    Ping $i: ${MS}ms"
done

# ── Done ──
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  DEPLOYMENT COMPLETE                                        ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║                                                              ║"
echo "║  COMMANDS:                                                   ║"
echo "║                                                              ║"
echo "║  Paper trade (default):                                      ║"
echo "║    sudo systemctl start polymarket-bot                       ║"
echo "║                                                              ║"
echo "║  Live trade (edit .env first!):                              ║"
echo "║    Edit /home/ubuntu/polymarket-pnl/.env                     ║"
echo "║    Then modify the service ExecStart to add --live           ║"
echo "║                                                              ║"
echo "║  Streaming signals only (no trading):                        ║"
echo "║    sudo systemctl start streaming-alpha                      ║"
echo "║                                                              ║"
echo "║  Monitor:                                                    ║"
echo "║    tail -f logs/bot.log                                      ║"
echo "║    sudo systemctl status polymarket-bot                      ║"
echo "║                                                              ║"
echo "║  Stop:                                                       ║"
echo "║    sudo systemctl stop polymarket-bot                        ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
