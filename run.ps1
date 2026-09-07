# One-command launcher: resolves the target RAG project's own venv Python
# and runs the eval loop with it -- no manual env vars or venv path typing.
# Forwards every argument straight to eval.runner, e.g.:
#   .\run.ps1
#   .\run.ps1 --num-answerable 50 --num-unanswerable 50
#   .\run.ps1 --rag-root D:\path\to\your-project
#
# Resolution order for the target project root, same as eval/target.py's:
#   1. RAG_PROJECT_ROOT environment variable, if set
#   2. This script's own directory -- the common case now: drop the eval\
#      folder plus this script directly into your RAG project's root and
#      run it from there, no env var needed at all.
#   3. A sibling directory named "RAG" next to this script's own folder --
#      backward compatible with running this suite as its own separate
#      repo, cloned alongside the target project.
# (--rag-root, passed through to eval.runner below, overrides all of this
# regardless of what this script picks.)
#
# This script deliberately does NOT check for any particular file (like
# app\config.py) inside whichever directory it picks -- a filename check
# is a proxy for "is this a compatible project" that breaks the moment
# someone's project is laid out differently (a flat main.py instead of an
# app\ package, say), and it can silently drift out of sync with what's
# actually required (this repo shipped exactly that bug once: this script
# kept checking for app\config.py after eval/target.py itself had already
# stopped requiring it). Real verification -- actually importing the
# target's embedder/generator modules and checking the required functions
# exist on them -- happens once, in Python, via eval.target.verify_target()
# when eval.runner starts. This script's only job is finding a directory
# with a virtualenv in it to hand off to.

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path

if ($env:RAG_PROJECT_ROOT) {
    $ragRoot = $env:RAG_PROJECT_ROOT
} elseif (Test-Path (Join-Path $here "backend\.venv\Scripts\python.exe")) {
    $ragRoot = Join-Path $here "backend"
} elseif (Test-Path (Join-Path $here ".venv\Scripts\python.exe")) {
    $ragRoot = $here
} else {
    $sibling = Join-Path (Split-Path -Parent $here) "RAG"
    if (Test-Path (Join-Path $sibling ".venv\Scripts\python.exe")) {
        $ragRoot = $sibling
    } else {
        $ragRoot = $here   # fall through to the "no virtualenv found" message below, naming this path
    }
}

if (-not (Test-Path $ragRoot -PathType Container)) {
    Write-Error @"
'$ragRoot' is not a directory.

Point at your project's root with:
  `$env:RAG_PROJECT_ROOT = "C:\path\to\your-project"
before running this script, or pass --rag-root <path> as an argument.
"@
    exit 1
}

$venvPython = Join-Path $ragRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Error @"
No virtualenv found at '$venvPython'.
Set up your project's venv first (python -m venv .venv, then .venv\Scripts\python.exe -m pip install -r requirements.txt inside it),
or point at a project that already has one with `$env:RAG_PROJECT_ROOT or --rag-root <path>.
"@
    exit 1
}

Write-Host "Target project: $ragRoot"
Write-Host "Using venv:     $venvPython"
Write-Host ""

Push-Location $here
try {
    & $venvPython -m eval.runner --rag-root $ragRoot @args
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
