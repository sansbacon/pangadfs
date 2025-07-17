#!/bin/bash

# Fetch and prune remote-tracking branches
echo "Fetching and pruning remote branches..."
git fetch -p

# Find local branches whose remote tracking branches are gone
branches_to_delete=$(git branch -vv | awk '/: gone]/{print $1}')

if [ -z "$branches_to_delete" ]; then
  echo "✅ No local branches to clean up."
  exit 0
fi

echo "The following local branches are tracking remotes that no longer exist:"
echo "$branches_to_delete"
echo

# Prompt for deletion
read -p "Do you want to delete these branches? (y/n): " confirm
if [[ "$confirm" =~ ^[Yy]$ ]]; then
  for branch in $branches_to_delete; do
    git branch -d "$branch" || {
      echo "⚠️ Could not delete $branch (not fully merged). Use -D to force."
    }
  done
else
  echo "❌ Cleanup canceled."
fi
