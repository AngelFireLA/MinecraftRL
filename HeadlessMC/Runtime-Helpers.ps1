function Resolve-LauncherJava {
    param([string]$RequestedPath)

    if ($RequestedPath) {
        if (-not (Test-Path -LiteralPath $RequestedPath -PathType Leaf)) {
            throw "Missing Java launcher at $RequestedPath."
        }
        return (Resolve-Path -LiteralPath $RequestedPath).Path
    }

    if ($env:JAVA_HOME) {
        $javaHomeExecutable = Join-Path $env:JAVA_HOME 'bin\java.exe'
        if (Test-Path -LiteralPath $javaHomeExecutable -PathType Leaf) {
            return $javaHomeExecutable
        }
    }

    $command = Get-Command java -CommandType Application -ErrorAction Stop | Select-Object -First 1
    return $command.Source
}

function Get-JarSha256 {
    param([Parameter(Mandatory = $true)][string]$Path)

    $resolvedPath = (Resolve-Path -LiteralPath $Path).Path
    $algorithm = [System.Security.Cryptography.SHA256]::Create()
    try {
        $stream = [System.IO.File]::OpenRead($resolvedPath)
        try {
            return [BitConverter]::ToString($algorithm.ComputeHash($stream)).Replace('-', '')
        }
        finally {
            $stream.Dispose()
        }
    }
    finally {
        $algorithm.Dispose()
    }
}
