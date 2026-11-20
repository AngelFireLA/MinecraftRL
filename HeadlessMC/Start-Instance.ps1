#Requires -Version 5.1

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^rl-[0-9]+$')]
    [string]$Name,

    [ValidateRange(384, 4096)]
    [int]$HeapMb = 1024,

    [ValidateRange(128, 4096)]
    [int]$InitialHeapMb = 256,

    [ValidateRange(1, 32)]
    [int]$ActiveProcessorCount = 4,

    [ValidateSet('off', 'event')]
    [string]$IdleWaitMode = 'event',

    [ValidateSet('default', 'g1')]
    [string]$GarbageCollector = 'default',

    [string]$GcLogPath,

    [ValidateRange(0, 100)]
    [int]$DisplaySleepMilliseconds = 0,

    [string]$MinecraftDirectory,
    [string]$LauncherJava,
    [switch]$RealAssets,
    [switch]$PauseOnFailure
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
Import-Module Microsoft.PowerShell.Utility -ErrorAction Stop
. (Join-Path $PSScriptRoot 'Runtime-Helpers.ps1')

function Get-MinecraftOfflineUuid {
    param([Parameter(Mandatory = $true)][string]$Username)

    $md5 = [System.Security.Cryptography.MD5]::Create()
    try {
        $bytes = [System.Text.Encoding]::UTF8.GetBytes("OfflinePlayer:$Username")
        $hash = $md5.ComputeHash($bytes)
    }
    finally {
        $md5.Dispose()
    }
    $hash[6] = [byte](($hash[6] -band 0x0f) -bor 0x30)
    $hash[8] = [byte](($hash[8] -band 0x3f) -bor 0x80)
    $hex = -join ($hash | ForEach-Object { $_.ToString('x2') })
    return '{0}-{1}-{2}-{3}-{4}' -f $hex.Substring(0, 8), $hex.Substring(8, 4),
        $hex.Substring(12, 4), $hex.Substring(16, 4), $hex.Substring(20, 12)
}

if (-not $MinecraftDirectory) {
    $MinecraftDirectory = Join-Path $PSScriptRoot 'shared\.minecraft'
}

$launcherJar = Join-Path $PSScriptRoot 'headlessmc-launcher.jar'
$instanceDirectory = Join-Path $PSScriptRoot "instances\$Name"
$metadataPath = Join-Path $instanceDirectory 'instance.json'
$builtModJar = Join-Path $PSScriptRoot '..\RLCraft\build\libs\minecraftrl-1.0.0.jar'
$instanceModJar = Join-Path $instanceDirectory 'mods\minecraftrl-1.0.0.jar'
$specificsTemplate = Join-Path $PSScriptRoot 'template\mods\hmc-specifics-1.8.9-2.4.0-lexforge-release.jar'
$specificsTarget = Join-Path $instanceDirectory 'mods\hmc-specifics-1.8.9-2.4.0-lexforge-release.jar'
$forgeVersion = '1.8.9-forge1.8.9-11.15.1.2318-1.8.9'
$forgeJson = Join-Path $MinecraftDirectory "versions\$forgeVersion\$forgeVersion.json"

if ($InitialHeapMb -gt $HeapMb) {
    throw "InitialHeapMb ($InitialHeapMb) cannot exceed HeapMb ($HeapMb)."
}
foreach ($requiredFile in @($launcherJar, $metadataPath, $builtModJar, $specificsTemplate)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw "Missing required file $requiredFile."
    }
}
if (-not (Test-Path -LiteralPath $forgeJson -PathType Leaf)) {
    throw "Missing Forge cache at $forgeJson.`nRun powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Initialize-Cache.ps1 first."
}

$metadata = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
if ($metadata.instance_id -ne $Name) {
    throw "Expected instance '$Name' but instance.json declares '$($metadata.instance_id)'."
}
$expectedOfflineUuid = Get-MinecraftOfflineUuid -Username $metadata.username
if ($metadata.uuid -ne $expectedOfflineUuid) {
    Write-Warning "Found unexpected UUID $($metadata.uuid) for $Name."
    Write-Warning "Using offline UUID $expectedOfflineUuid for $($metadata.username)."
    $metadata.uuid = $expectedOfflineUuid
    [System.IO.File]::WriteAllText(
        $metadataPath,
        ($metadata | ConvertTo-Json) + [Environment]::NewLine,
        [System.Text.UTF8Encoding]::new($false)
    )
}

New-Item -ItemType Directory -Path (Split-Path -Parent $instanceModJar) -Force | Out-Null
$copyMod = -not (Test-Path -LiteralPath $instanceModJar -PathType Leaf)
if (-not $copyMod) {
    $copyMod = (Get-JarSha256 $builtModJar) -ne (Get-JarSha256 $instanceModJar)
}
if ($copyMod) {
    Copy-Item -LiteralPath $builtModJar -Destination $instanceModJar -Force
    Write-Host "Updated the MinecraftRL mod in $Name."
}
if (-not (Test-Path -LiteralPath $specificsTarget -PathType Leaf)) {
    Copy-Item -LiteralPath $specificsTemplate -Destination $specificsTarget
}

$resolvedLauncherJava = Resolve-LauncherJava -RequestedPath $LauncherJava
$normalizedMinecraftDirectory = (Resolve-Path -LiteralPath $MinecraftDirectory).Path.Replace('\', '/')
$normalizedInstanceDirectory = (Resolve-Path -LiteralPath $instanceDirectory).Path.Replace('\', '/')
$launcherRuntimeDirectory = Join-Path $instanceDirectory '.headlessmc-runtime'
New-Item -ItemType Directory -Path $launcherRuntimeDirectory -Force | Out-Null
$dummyAssets = (-not $RealAssets.IsPresent).ToString().ToLowerInvariant()
$gameJvmArgumentList = @(
    "-Xms${InitialHeapMb}m"
    "-Xmx${HeapMb}m"
    "-XX:ActiveProcessorCount=$ActiveProcessorCount"
    '-Djava.awt.headless=true'
    "-Dminecraftrl.idleWait=$IdleWaitMode"
    "-Dminecraftrl.instanceId=$($metadata.instance_id)"
    "-Dminecraftrl.port=$($metadata.port)"
    "-Dminecraftrl.world=$($metadata.world)"
)
if ($GcLogPath) {
    $normalizedGcLogPath = [System.IO.Path]::GetFullPath($GcLogPath).Replace('\', '/')
    $gameJvmArgumentList += @(
        '-XX:+PrintGCDetails'
        '-XX:+PrintGCDateStamps'
        "-Xloggc:$normalizedGcLogPath"
    )
}
if ($GarbageCollector -eq 'g1') {
    $gameJvmArgumentList += '-XX:+UseG1GC'
}
$gameJvmArguments = $gameJvmArgumentList -join ' '

$launcherArguments = @(
    "-Dhmc.mcdir=$normalizedMinecraftDirectory",
    "-Dhmc.gamedir=$normalizedInstanceDirectory",
    '-Dhmc.offline=true',
    "-Dhmc.offline.username=$($metadata.username)",
    "-Dhmc.offline.uuid=$($metadata.uuid)",
    "-Dhmc.assets.dummy=$dummyAssets",
    '-Dhmc.filehandler.enabled=false',
    '-Dhmc.java.require.exact=true',
    "-Dhmc.lwjgl.update_sleep=$DisplaySleepMilliseconds",
    "-Dhmc.jvmargs=$gameJvmArguments",
    '-jar',
    $launcherJar,
    '--command',
    "launch $forgeVersion -lwjgl -paulscode -offline"
)

Write-Host "Launching $Name (port=$($metadata.port), heap=${HeapMb}MiB, active-processors=$ActiveProcessorCount, gc=$GarbageCollector, display-sleep=${DisplaySleepMilliseconds}ms, dummy-assets=$dummyAssets)"
Write-Host "gameDir=$instanceDirectory"
Write-Host "mcDir=$MinecraftDirectory"
Write-Host "launcherRuntime=$launcherRuntimeDirectory"
Write-Host "Use 'gui' to inspect the HeadlessMC-Specifics screen and 'quit' to stop the client."

$launcherExitCode = 1
Push-Location $launcherRuntimeDirectory
try {
    & $resolvedLauncherJava @launcherArguments
    $launcherExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
if ($launcherExitCode -ne 0 -and $PauseOnFailure.IsPresent) {
    Write-Host "HeadlessMC for $Name exited with code $launcherExitCode." -ForegroundColor Red
    [void](Read-Host 'Press Enter to close this window')
}
exit $launcherExitCode
