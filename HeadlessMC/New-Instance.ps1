#Requires -Version 5.1

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^rl-[0-9]+$')]
    [string]$Name,

    [ValidateRange(0, 65535)]
    [int]$Port = 0,

    [string]$WorldTemplate,
    [string]$MinecraftRlJar,
    [string]$SpecificsJar
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-MinecraftOfflineUuid {
    param([Parameter(Mandatory = $true)][string]$Username)

    # Minecraft's offline-mode UUID is UUID.nameUUIDFromBytes("OfflinePlayer:<name>").
    # Formatting the MD5 bytes ourselves avoids System.Guid's mixed-endian byte layout.
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

if (-not $MinecraftRlJar) {
    $MinecraftRlJar = Join-Path $PSScriptRoot '..\RLCraft\build\libs\minecraftrl-1.0.0.jar'
}
if (-not $SpecificsJar) {
    $SpecificsJar = Join-Path $PSScriptRoot 'template\mods\hmc-specifics-1.8.9-2.4.0-lexforge-release.jar'
}

if ($Name -notmatch '^rl-([0-9]+)$') {
    throw "Invalid instance name '$Name'."
}
$index = [int]$Matches[1]
if ($Port -eq 0) {
    $Port = 8765 + $index
}
if ($Port -gt 65535) {
    throw "Calculated port $Port exceeds 65535."
}

$instanceDirectory = Join-Path $PSScriptRoot "instances\$Name"
$optionsTemplate = Join-Path $PSScriptRoot 'template\options.txt'
$splashTemplate = Join-Path $PSScriptRoot 'template\splash.properties'

foreach ($requiredFile in @($MinecraftRlJar, $SpecificsJar, $optionsTemplate, $splashTemplate)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw "Missing required file $requiredFile."
    }
}
if ($WorldTemplate -and -not (Test-Path -LiteralPath (Join-Path $WorldTemplate 'level.dat') -PathType Leaf)) {
    throw "World template is missing level.dat: $WorldTemplate"
}
if (Test-Path -LiteralPath $instanceDirectory) {
    throw "An instance already exists at $instanceDirectory. Choose another name."
}

$modsDirectory = Join-Path $instanceDirectory 'mods'
$savesDirectory = Join-Path $instanceDirectory 'saves'
$configDirectory = Join-Path $instanceDirectory 'config'
$worldDirectory = Join-Path $savesDirectory 'void'
if (-not $WorldTemplate) {
    $python = (Get-Command python -CommandType Application -ErrorAction Stop | Select-Object -First 1).Source
}
New-Item -ItemType Directory -Path $modsDirectory, $savesDirectory, $configDirectory -Force | Out-Null

if ($WorldTemplate) {
    Copy-Item -LiteralPath $WorldTemplate -Destination $worldDirectory -Recurse
}
else {
    & $python (Join-Path $PSScriptRoot 'template\create_void_world.py') $worldDirectory
    if ($LASTEXITCODE -ne 0) {
        throw "Could not generate the void world for $Name (Python exited with $LASTEXITCODE)."
    }
}
Copy-Item -LiteralPath $MinecraftRlJar -Destination $modsDirectory
Copy-Item -LiteralPath $SpecificsJar -Destination $modsDirectory
Copy-Item -LiteralPath $optionsTemplate -Destination (Join-Path $instanceDirectory 'options.txt')
Copy-Item -LiteralPath $splashTemplate -Destination (Join-Path $configDirectory 'splash.properties')

$username = "RLAgent$index"
$metadata = [ordered]@{
    instance_id = $Name
    index = $index
    port = $Port
    world = 'void'
    username = $username
    uuid = Get-MinecraftOfflineUuid -Username $username
}
$metadataJson = $metadata | ConvertTo-Json
$metadataPath = Join-Path $instanceDirectory 'instance.json'
[System.IO.File]::WriteAllText(
    $metadataPath,
    $metadataJson + [Environment]::NewLine,
    [System.Text.UTF8Encoding]::new($false)
)

Write-Host "Created instance at $instanceDirectory."
Write-Host "instance=$Name username=$($metadata.username) uuid=$($metadata.uuid) port=$Port world=void"
