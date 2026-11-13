#!/bin/bash
# Clone the authors' code of every baseline at the commit the paper used, into $THIRD_PARTY (default: third_party/).
# Usage: baselines/fetch.sh [name ...]     (no name: every repository)
# No baseline needs a patch to its source: the wrappers in baselines/ import the authors' modules unchanged.
set -euo pipefail
BASELINES=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); THIRD_PARTY=${THIRD_PARTY:-$BASELINES/../third_party}
REPOS=(  # name (folder under $THIRD_PARTY)  URL  commit
  "hrm/HRM     https://github.com/sapientinc/HRM                           ac15626f8db096a63c775b84c9dc868776a6feda"
  "trm         https://github.com/SamsungSAILMontreal/TinyRecursiveModels  c01103738605ba39d1430519b1ee0c62f4c707f8"
  "gram        https://github.com/ahn-ml/gram-release                      f2e32b9b11737890bfd4f39ead4cbea90835f91b"
  "eqr         https://github.com/locuslab/EqR                             134c7541711125860954c78463023c583b2ef21b"
  "frm         https://github.com/helblazer811/Flow-Reasoning-Models       6149a01d529ab312366a67bf2b178bfa6d54c3ff"
  "ebr         https://github.com/wwillchen/ebr                            8c5271c05cf35f816e36db41ff324b3081175c64"
  "fprm/fprm   https://github.com/nilskiKonjIzDunava/fprm                  b291252387b2656d7a7bc2719d9317e6d812b283"
)
mkdir -p "$THIRD_PARTY"
for entry in "${REPOS[@]}"; do
  read -r name url commit <<< "$entry"
  if [ $# -gt 0 ] && [[ ! " $* " == *" $name "* ]]; then continue; fi
  dst=$THIRD_PARTY/$name
  if [ ! -d "$dst/.git" ]; then
    mkdir -p "$(dirname "$dst")"
    git clone --quiet "$url" "$dst"
  fi
  git -C "$dst" fetch --quiet origin "$commit" 2>/dev/null || true
  git -C "$dst" checkout --quiet --detach "$commit"
  [ "$(git -C "$dst" rev-parse HEAD)" = "$commit" ] || { echo "$name: not at $commit"; exit 1; }
  echo "$name at ${commit:0:7}"
done
