# LlamaBot SaaS – Staging Deployment Guide (Ubuntu 20 LTS)

> **Audience:** new engineers & AI agents tasked with standing up, updating, or recovering a LlamaBot staging box.
> **Scope:** single‑VM install on AWS (Lightsail/EC2), Python 3.11, Postgres checkpointing, HTTPS & WSS via Nginx.

---

## 1  Provision the VM

| Step | Command / Action                                                                                       |
| ---- | ------------------------------------------------------------------------------------------------------ |
| 1.1  | Launch Ubuntu 20.04 LTS (minimum 2 vCPU / 2 GB). Attach a static IP.                                   |
| 1.2  | In the provider’s firewall open **TCP 22, 80, 443**. Postgres (**5432**) stays closed until Section 5. |
| 1.3  | SSH key‑auth only. Add your public key to `~/.ssh/authorized_keys`.                                    |

---

## 2  Base toolchain

```bash
sudo apt update && sudo apt install -y \
  python3-venv build-essential curl git nginx \
  libssl-dev zlib1g-dev libpq5

# Python 3.11 via pyenv
curl https://pyenv.run | bash
exec $SHELL                    # reload PATH
pyenv install 3.11.9
pyenv global 3.11.9
```

---

## 3  Clone & install LlamaBot

```bash
cd /home/ubuntu
git clone git@github.com:llamapressai/LlamaBotSaaS.git LlamaBot
cd LlamaBot
python -m venv venv && source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt            # incl. playwright
playwright install-deps && playwright install
pip install -e .                            # registers the `app` package
```

### 3.1  Environment file

```bash
cp .env.example .env        # then fill in ↓
OPENAI_API_KEY=sk‑…
DB_URI=postgresql://<user>:<pass>@<db‑ip>:5432/<db>
LLAMAPRESS_API_URL=https://<rails‑host>
```

---

## 4  App daemon (systemd)

```ini
[Unit]
Description=LlamaBot SaaS (Uvicorn)
After=network.target

[Service]
User=ubuntu
Group=www-data
WorkingDirectory=/home/ubuntu/LlamaBot
ExecStart=/home/ubuntu/LlamaBot/venv/bin/uvicorn app.main:app \
          --host 0.0.0.0 --port 8000 --workers 4
Restart=on-failure
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now llamabot
```

---

## 5  Postgres (HatchBox‑provisioned)

The Rails side (LlamaPress) spins up its own Postgres instance automatically during deployment, so **you never create or init the database on this VM.** Instead you only fetch the private connection string and drop it into `.env`.

1. **Grab the connection URI**
   HatchBox ▶ *Your App* ▶ **Databases** ▶ click the DB → copy **Private Connection URI** (e.g. `postgresql://user:pass@10.x.x.x:5432/db`).
   ↳ Full Rails deployment notes are in the HatchBox guide: [https://www.notion.so/LlamaPress-HatchBox-Deployment-1353f4768d3a80798c6fe2eedc8b9ecc](https://www.notion.so/LlamaPress-HatchBox-Deployment-1353f4768d3a80798c6fe2eedc8b9ecc).
2. **Wire LlamaBot to that DB**

   ```bash
   cd /home/ubuntu/LlamaBot
   nano .env                  # or vi
   # add / update:
   DB_URI=<paste-the-URI>
   ```
3. **Firewall**
   In HatchBox/the cloud console, allow inbound **TCP 5432** from the LlamaBot VM’s public IP. No ufw change is needed on LlamaBot.
4. **Connectivity & checkpoint tables**

   ```bash
   source venv/bin/activate
   psql "$DB_URI" -c 'select 1;'     # expect one row
   python app/init_pg_checkpointer.py  # prints ✅ tables & indexes initialised
   deactivate
   ```
5. **Restart the service**

   ```bash
   sudo systemctl restart llamabot
   sudo systemctl status  llamabot --no-pager -n 10  # should stay active
   ```

---

## 6  Nginx + Certbot + WebSocket  Nginx + Certbot + WebSocket

```bash
sudo tee /etc/nginx/sites-available/llamabot <<'EOF'
server {
    listen 80;
    server_name <your-llamabot-domain>;
    location / {
        proxy_pass http://unix:/home/ubuntu/LlamaBot/gunicorn.sock;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
EOF
sudo ln -s /etc/nginx/sites-available/llamabot /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

# TLS
sudo apt-get install -y certbot python3-certbot-nginx
sudo certbot --nginx -d <your-llamabot-domain> --redirect
```

---

## 7  Smoke tests

```bash
# HTTPS
curl -I https://<your-llamabot-domain>/hello  # 200/JSON

# WebSocket (install once: sudo npm i -g wscat)
wscat -c wss://<your-llamabot-domain>/ws      # expect 101 switch
> {"ping":"hi"}
< {"pong":"hi"}
```

---

## 8  Day‑2 operations

| Task                          | Command                                                                          |
| ----------------------------- | -------------------------------------------------------------------------------- |
| **Deploy code update**        | `git pull && pip install -r requirements.txt && sudo systemctl restart llamabot` |
| **Logs**                      | `sudo journalctl -u llamabot -n 100 --no-pager`                                  |
| **Follow Nginx access/error** | `tail -f /var/log/nginx/*.log`                                                   |
| **Renew TLS**                 | `sudo certbot renew --dry-run`                                                   |
| **Postgres backups**          | use `pg_dump` from LlamaPress host or managed service snapshot                   |

---

✨  *LlamaBot staging box is now live with Python 3.11, HTTPS & WSS, Postgres checkpointing, and CI‑friendly import paths.*
