#!/bin/bash
# =============================================================
# AWS EC2 Deployment Script for Streaming Alpha Engine
# =============================================================
#
# RECOMMENDED INSTANCE:
#   Region:    ap-northeast-1 (Tokyo) — closest to Binance
#   Instance:  c5.large (2 vCPU, 4GB RAM) for production
#              t3.medium (2 vCPU, 4GB RAM) for testing
#   AMI:       Ubuntu 24.04 LTS (ami-0a5a6128c1707a2d3)
#   Storage:   20GB gp3
#
# EXPECTED LATENCY:
#   From Tokyo EC2 to Binance API: ~2-5ms
#   From US East to Binance API:   ~150-200ms
#   From Europe to Binance API:    ~100-150ms
#
# COST:
#   t3.medium:  ~$30/month
#   c5.large:   ~$62/month
#
# USAGE:
#   1. Launch EC2 in ap-northeast-1
#   2. SSH into the instance
#   3. Run: bash deploy_aws.sh
# =============================================================

set -euo pipefail

echo "============================================"
echo "  Deploying Streaming Alpha Engine"
echo "============================================"

# ── System updates ──
echo "[1/6] Updating system..."
sudo apt-get update -qq
sudo apt-get upgrade -y -qq

# ── Python 3.11+ ──
echo "[2/6] Installing Python..."
sudo apt-get install -y -qq python3 python3-pip python3-venv git

# ── Clone repo ──
echo "[3/6] Cloning repository..."
cd /home/ubuntu
if [ -d "polymarket-pnl" ]; then
    cd polymarket-pnl && git pull
else
    git clone https://github.com/EZASI/polymarket-pnl.git
    cd polymarket-pnl
fi

# ── Create virtual environment ──
echo "[4/6] Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r deploy/requirements.txt

# ── Create directories ──
mkdir -p logs models

# ── Install systemd service ──
echo "[5/6] Installing systemd service..."
sudo cp deploy/streaming.service /etc/systemd/system/streaming-alpha.service

# Update service to use venv python
sudo sed -i "s|/usr/bin/python3|/home/ubuntu/polymarket-pnl/venv/bin/python3|g" \
    /etc/systemd/system/streaming-alpha.service

sudo systemctl daemon-reload
sudo systemctl enable streaming-alpha.service

# ── Test latency ──
echo "[6/6] Testing Binance API latency..."
for i in 1 2 3; do
    LATENCY=$(curl -o /dev/null -s -w '%{time_total}\n' https://api.binance.com/api/v3/ping)
    echo "  Ping $i: ${LATENCY}s"
done

echo ""
echo "============================================"
echo "  Deployment Complete!"
echo "============================================"
echo ""
echo "  Commands:"
echo "    Start:   sudo systemctl start streaming-alpha"
echo "    Stop:    sudo systemctl stop streaming-alpha"
echo "    Status:  sudo systemctl status streaming-alpha"
echo "    Logs:    tail -f logs/streaming.log"
echo ""
echo "  First run will self-train on 7 days of data."
echo "  Predictions start ~2 minutes after boot."
echo ""
echo "  To test manually:"
echo "    source venv/bin/activate"
echo "    python3 streaming_predictor.py --self-train"
echo ""
