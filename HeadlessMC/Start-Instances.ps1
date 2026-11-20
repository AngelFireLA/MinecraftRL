#Requires -Version 5.1

[CmdletBinding()]
param(
    [ValidateRange(1, 16)]
    [int]$Count = 2,

    [ValidateRange(0, 1000)]
    [int]$FirstIndex = 0,

    [ValidateRange(384, 4096)]
    [int]$HeapMb = 1024,

    [ValidateRange(1, 32)]
    [int]$ActiveProcessorCount = 4,

    [ValidateSet('off', 'event')]
    [string]$IdleWaitMode = 'event',

    [ValidateSet('default', 'g1')]
    [string]$GarbageCollector = 'default',

    [ValidateRange(0, 100)]
    [int]$DisplaySleepMilliseconds = 0,

    [ValidateRange(0, 15)]
    [int]$StaggerSeconds = 2
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$startScript = Join-Path $PSScriptRoot 'Start-Instance.ps1'
for ($offset = 0; $offset -lt $Count; $offset++) {
    $index = $FirstIndex + $offset
    $name = "rl-$index"
    $instanceDirectory = Join-Path $PSScriptRoot "instances\$name"
    if (-not (Test-Path -LiteralPath $instanceDirectory -PathType Container)) {
        throw "Missing instance $name. Create it with .\New-Instance.ps1 -Name $name."
    }

    $arguments = @(
        '-NoProfile',
        '-ExecutionPolicy', 'Bypass',
        '-File', ('"{0}"' -f $startScript),
        '-Name', $name,
        '-HeapMb', $HeapMb,
        '-ActiveProcessorCount', $ActiveProcessorCount,
        '-GarbageCollector', $GarbageCollector,
        '-IdleWaitMode', $IdleWaitMode,
        '-DisplaySleepMilliseconds', $DisplaySleepMilliseconds,
        '-PauseOnFailure'
    )
    Start-Process -FilePath 'powershell.exe' -ArgumentList $arguments -WorkingDirectory $PSScriptRoot -WindowStyle Normal
    Write-Host "Opened a client console for $name."

    if ($StaggerSeconds -gt 0 -and $offset + 1 -lt $Count) {
        Start-Sleep -Seconds $StaggerSeconds
    }
}
