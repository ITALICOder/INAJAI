import os
import asyncio
import signal
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.live import Live

# Configuration
START_PORT = 6990
END_PORT = 7000
BASE_DIR = Path.home() / "tor_multi"
DATA_DIR = BASE_DIR / "data"
EMPTY_CONF = BASE_DIR / "empty.conf"

console = Console()

class TorProxy:
    def __init__(self, port):
        self.port = port
        self.process = None
        self.log_buffer = []

    async def start(self):
        data_path = DATA_DIR / f"data_{self.port}"
        data_path.mkdir(parents=True, exist_ok=True)
        # Create empty config if not exists
        if not EMPTY_CONF.exists():
            EMPTY_CONF.parent.mkdir(parents=True, exist_ok=True)
            EMPTY_CONF.write_text("# empty config\n")
        cmd = [
            "tor",
            # Ignore default settings
            "--defaults-torrc", str(EMPTY_CONF),
            "-f", str(EMPTY_CONF),
            "--ignore-missing-torrc",
            # Proxy settings
            "--SocksPort", str(self.port),
            "--CookieAuthentication", "0",
            "--DataDirectory", str(data_path),
            "--Log", "notice stdout"
        ]
        # Launch tor process
        self.process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
        )
        asyncio.create_task(self._read_logs())

    async def stop(self):
        if self.process and self.process.returncode is None:
            self.process.send_signal(signal.SIGINT)
            await self.process.wait()

    async def _read_logs(self):
        assert self.process.stdout
        async for line in self.process.stdout:
            text = line.decode().rstrip()
            self.log_buffer.append(text)
            if len(self.log_buffer) > 5:
                self.log_buffer.pop(0)

    def status(self):
        if self.process is None:
            return "stopped"
        return "running" if self.process.returncode is None else "exited"

    def pid(self):
        return str(self.process.pid) if self.process else "-"

    def recent_logs(self):
        return "\n".join(self.log_buffer)

async def run_manager():
    # Prepare directories and empty config
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    proxies = [TorProxy(port) for port in range(START_PORT, END_PORT + 1)]

    # Start all proxies
    await asyncio.gather(*(p.start() for p in proxies))

    console.clear()
    console.show_cursor(False)

    def make_table():
        table = Table(title="Tor SOCKS5 Proxies")
        table.add_column("Port", style="cyan")
        table.add_column("Status", style="magenta")
        table.add_column("PID", style="green")
        table.add_column("Recent Logs", style="yellow")
        for p in proxies:
            table.add_row(str(p.port), p.status(), p.pid(), p.recent_logs())
        return table

    with Live(make_table(), refresh_per_second=1, console=console) as live:
        try:
            while True:
                live.update(make_table())
                await asyncio.sleep(1)
        except (KeyboardInterrupt, asyncio.CancelledError):
            console.print("\nStopping all proxies...", style="bold red")
            await asyncio.gather(*(p.stop() for p in proxies))
            console.show_cursor(True)
            sys.exit(0)

if __name__ == "__main__":
    try:
        asyncio.run(run_manager())
    except KeyboardInterrupt:
        pass
