$PythonExe = "c:\Users\willi\anaconda3\envs\fitdata\python.exe"
Write-Host "Running tests using: $PythonExe"
& $PythonExe -m pytest @args
