#Requires -Version 5.1

[CmdletBinding()]
param(
    [string]$MinecraftDirectory,
    [string]$LauncherJava
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Runtime-Helpers.ps1')

if (-not $MinecraftDirectory) {
    $MinecraftDirectory = Join-Path $PSScriptRoot 'shared\.minecraft'
}

$launcherJar = Join-Path $PSScriptRoot 'headlessmc-launcher.jar'
$vanillaJson = Join-Path $MinecraftDirectory 'versions\1.8.9\1.8.9.json'
$forgeVersion = '1.8.9-forge1.8.9-11.15.1.2318-1.8.9'
$forgeJson = Join-Path $MinecraftDirectory "versions\$forgeVersion\$forgeVersion.json"

function Install-ReleaseJar {
    param([string]$Url, [string]$Path, [string]$Sha256)

    if (Test-Path -LiteralPath $Path -PathType Leaf) {
        return
    }

    $directory = Split-Path -Parent $Path
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
    $temporaryJar = Join-Path $directory ([guid]::NewGuid().ToString() + '.download')
    try {
        Write-Host "Downloading $(Split-Path -Leaf $Path)"
        Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $temporaryJar
        if ((Get-JarSha256 $temporaryJar) -ne $Sha256) {
            throw "SHA-256 mismatch for $Url"
        }
        try {
            [System.IO.File]::Move($temporaryJar, $Path)
        }
        catch [System.IO.IOException] {
            if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
                throw
            }
        }
    }
    finally {
        if (Test-Path -LiteralPath $temporaryJar -PathType Leaf) {
            Remove-Item -LiteralPath $temporaryJar
        }
    }
}

function Invoke-HeadlessMcCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command
    )

    $normalizedMinecraftDirectory = $MinecraftDirectory.Replace('\', '/')
    $arguments = @(
        "-Dhmc.mcdir=$normalizedMinecraftDirectory",
        '-Dhmc.filehandler.enabled=false',
        '-Dhmc.exit.on.failed.command=true',
        '-jar',
        $launcherJar,
        '--command',
        $Command
    )

    Write-Host "HeadlessMC> $Command"
    & $script:ResolvedLauncherJava @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "HeadlessMC exited with code $LASTEXITCODE while running '$Command'."
    }
}

$script:ResolvedLauncherJava = Resolve-LauncherJava -RequestedPath $LauncherJava
$specificsJar = 'hmc-specifics-1.8.9-2.4.0-lexforge-release.jar'
Install-ReleaseJar `
    -Url 'https://github.com/headlesshq/headlessmc/releases/download/2.10.0/headlessmc-launcher-2.10.0.jar' `
    -Path $launcherJar `
    -Sha256 '52bd5006f478377b3893011d458562977d38c65ead6d2b31089beb4d614f13cd'
Install-ReleaseJar `
    -Url "https://github.com/headlesshq/hmc-specifics/releases/download/2.4.0/$specificsJar" `
    -Path (Join-Path $PSScriptRoot "template\mods\$specificsJar") `
    -Sha256 'b5eebe93e106ecc9ed8c6480212728e22e2e6b984f07536f778f25b195d87e84'

New-Item -ItemType Directory -Path $MinecraftDirectory -Force | Out-Null

Push-Location $PSScriptRoot
try {
    if (Test-Path -LiteralPath $vanillaJson -PathType Leaf) {
        Write-Host 'Minecraft 1.8.9 is already in the shared cache.'
    }
    else {
        Invoke-HeadlessMcCommand -Command 'download 1.8.9'
    }

    if (Test-Path -LiteralPath $forgeJson -PathType Leaf) {
        Write-Host "Forge $forgeVersion is already in the shared cache."
    }
    else {
        Invoke-HeadlessMcCommand -Command 'forge 1.8.9 --uid 11.15.1.2318'
    }
}
finally {
    Pop-Location
}

Write-Host "HeadlessMC cache is ready at $MinecraftDirectory."
Write-Host 'The first launch downloads any missing libraries and dummy assets.'
