$ErrorActionPreference = 'Stop'

$debugAddress = '100.121.175.55'
$debugPort = 9222
$profileDir = 'E:\playwright-profile'
$logDir = 'E:\logs'
$browserLog = Join-Path $logDir 'playwright-chromium.log'
$proxyScript = Join-Path $logDir 'playwright-cdp-host-proxy.js'
$proxyLog = Join-Path $logDir 'playwright-cdp-host-proxy.log'
$playwrightBrowsersPath = 'E:\ms-playwright'

New-Item -ItemType Directory -Force -Path $profileDir | Out-Null
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

try {
  netsh interface portproxy delete v4tov4 listenaddress=$debugAddress listenport=$debugPort | Out-Null
  if (-not (Get-NetFirewallRule -DisplayName 'OSS Playwright CDP 9222' -ErrorAction SilentlyContinue)) {
    New-NetFirewallRule -DisplayName 'OSS Playwright CDP 9222' -Direction Inbound -Action Allow -Protocol TCP -LocalAddress $debugAddress -LocalPort $debugPort | Out-Null
  }
} catch {
  Write-Warning "Could not configure $debugAddress`:$debugPort firewall rule: $($_.Exception.Message)"
}

$chromium = Get-ChildItem -Path $playwrightBrowsersPath -Filter chrome.exe -Recurse -ErrorAction SilentlyContinue |
  Where-Object { $_.FullName -match '\\chromium-[^\\]+\\chrome-win64?\\chrome\.exe$' } |
  Sort-Object FullName -Descending |
  Select-Object -First 1

if (-not $chromium) {
  throw "Could not find Playwright Chromium under $playwrightBrowsersPath. Run 'npx playwright install chromium' with PLAYWRIGHT_BROWSERS_PATH=$playwrightBrowsersPath."
}

function Quote-CmdArg {
  param([Parameter(Mandatory = $true)][string]$Value)
  '"' + ($Value -replace '"', '\"') + '"'
}

$chromePath = $chromium.FullName
$chromeArgs = @(
  "--remote-debugging-address=$debugAddress",
  "--remote-debugging-port=$debugPort",
  "--user-data-dir=$profileDir",
  '--enable-unsafe-webgpu',
  '--enable-features=Vulkan',
  '--no-first-run',
  '--no-default-browser-check',
  '--disable-features=Translate',
  '--disable-translate',
  '--remote-allow-origins=*',
  'about:blank'
)

$stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
Add-Content -Path $browserLog -Value "[$stamp] launching $chromePath on $debugAddress`:$debugPort"

$cmd = 'cmd /c "' + (Quote-CmdArg $chromePath) + ' ' + ($chromeArgs -join ' ') + ' >> ' + (Quote-CmdArg $browserLog) + ' 2>&1"'
$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd }

if ($result.ReturnValue -ne 0) {
  throw "Win32_Process.Create failed with ReturnValue=$($result.ReturnValue)"
}

$proxy = @"
const net = require("node:net");

const listenHost = "$debugAddress";
const listenPort = $debugPort;
const upstreamHost = "127.0.0.1";
const upstreamPort = $debugPort;
const hostHeader = "Host: $debugAddress`:$debugPort";

function rewriteHost(chunk) {
  const text = chunk.toString("latin1");
  const end = text.indexOf("\r\n\r\n");
  if (end === -1) return chunk;
  const head = text.slice(0, end).replace(/^Host:.*$/im, hostHeader);
  return Buffer.from(head + text.slice(end), "latin1");
}

const server = net.createServer((client) => {
  const upstream = net.connect(upstreamPort, upstreamHost);
  let first = true;

  client.on("data", (chunk) => {
    upstream.write(first ? rewriteHost(chunk) : chunk);
    first = false;
  });
  upstream.on("data", (chunk) => client.write(chunk));
  client.on("error", () => upstream.destroy());
  upstream.on("error", () => client.destroy());
  client.on("close", () => upstream.destroy());
  upstream.on("close", () => client.destroy());
});

server.listen(listenPort, listenHost);

setInterval(() => {
  const probe = net.connect(upstreamPort, upstreamHost);
  probe.on("connect", () => probe.end());
  probe.on("error", () => process.exit(2));
}, 10000).unref();
"@

Set-Content -Path $proxyScript -Value $proxy -Encoding Ascii
$proxyAlive = Get-CimInstance Win32_Process -Filter "name = 'node.exe'" -ErrorAction SilentlyContinue |
  Where-Object { $_.CommandLine -match [regex]::Escape($proxyScript) }

if (-not $proxyAlive) {
  $nodePath = (Get-Command node.exe -ErrorAction Stop).Source
  $proxyCmd = 'cmd /c "' + (Quote-CmdArg $nodePath) + ' ' + (Quote-CmdArg $proxyScript) + ' >> ' + (Quote-CmdArg $proxyLog) + ' 2>&1"'
  Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $proxyCmd } | Out-Null
}

$result | Select-Object ProcessId, ReturnValue
