#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 [gemini|claude]"
    exit 1
fi

MODEL=$1

if [ "$MODEL" != "gemini" ] && [ "$MODEL" != "claude" ]; then
    echo "Invalid model specified. Use either 'gemini' or 'claude'"
    exit 1
fi

read -r -d '' PROMPT << 'EOF'
You are a senior software engineer on the Octogen team.
You goal is to maintain a code base with no Python type errors or test errors.

Please perform the following tasks:
Type checking
* Run the script ./tests/typecheck.sh. This script will run mypy type checking.
* Examine the output of the script and fix the reported mypy type errors.
* While fixing the mypy type errors, please try as much as possible to avoid introducing the Any type or ignore comments.
* If a library is missing type stubs or py.typed file, then please add it to the section `[[tool.mypy.overrides]]` in the root pyproject.toml file.

Focus on maintaining code quality and ensuring all tests pass.
If you unable to fix the issues after 10 attempts, then please stop, write a short summary of the issues and the steps you took to fix them, and then exit.
EOF

if [ "$MODEL" == "gemini" ]; then
    gemini --yolo --prompt "$PROMPT"
else
    claude "$PROMPT" --allowedTools "Bash(uv*),Bash(git*),Bash(./tests/typecheck.sh)"
fi
