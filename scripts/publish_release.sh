#!/usr/bin/env bash
# Publish a built release tree to the deployment branch (default: streamlit-release).
# Appends one commit per release so the branch history is the deployment history.
#
# Env: REMOTE_URL (git remote), SHA (source commit), TREE (dir from `release build`),
#      BRANCH (optional)
set -euo pipefail

: "${REMOTE_URL:?REMOTE_URL is required}"
: "${SHA:?SHA is required}"
: "${TREE:?TREE is required}"
BRANCH="${BRANCH:-streamlit-release}"
WORK="$(mktemp -d)"

if git ls-remote --exit-code --heads "$REMOTE_URL" "$BRANCH" >/dev/null 2>&1; then
  git clone --quiet --depth 1 --branch "$BRANCH" "$REMOTE_URL" "$WORK"
else
  git -C "$WORK" init --quiet -b "$BRANCH"
  git -C "$WORK" remote add origin "$REMOTE_URL"
fi

# replace the whole tree so files removed from the release disappear from the branch
find "$WORK" -mindepth 1 -maxdepth 1 -not -name .git -exec rm -rf {} +
cp -a "$TREE"/. "$WORK"/

cd "$WORK"
git config user.name  "${GIT_AUTHOR_NAME:-github-actions[bot]}"
git config user.email "${GIT_AUTHOR_EMAIL:-41898282+github-actions[bot]@users.noreply.github.com}"
git add -A
if git diff --cached --quiet; then
  echo "Release tree identical to $BRANCH; nothing to publish."
  exit 0
fi
git commit --quiet -m "Release ${SHA}"
git push --quiet origin "HEAD:refs/heads/${BRANCH}"
echo "Published $(git rev-parse --short HEAD) to ${BRANCH} (source ${SHA:0:7})"
