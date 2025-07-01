#! /bin/bash

if ! GIT_ROOT=$(git rev-parse --show-toplevel); then
    echo "Not in a git repository"
    exit 1
fi

pushd "${GIT_ROOT}" > /dev/null || exit 1
uv run mypy --explicit-package-bases --config-file "${GIT_ROOT}/pyproject.toml" src
MYPY_EXIT_CODE_1=$?
popd > /dev/null || exit 1

# Exit with error if any mypy check failed
if [ $MYPY_EXIT_CODE_1 -ne 0 ]; then
    exit 1
fi

exit 0
