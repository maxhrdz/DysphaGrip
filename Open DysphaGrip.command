#!/bin/zsh
cd "${0:A:h}" || exit 1
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
if [[ -x .venv/bin/python ]]; then
  exec .venv/bin/python local_dashboard.py
fi
exec /opt/anaconda3/bin/python local_dashboard.py
