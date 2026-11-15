# Taken from scripts/tp4.sh of github.com/MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104 (function node_inventory),
# Copyright 2026 Mia's AI Lab (MiaAI-Lab), licensed under the Apache License, Version 2.0.
# Changes: extracted into a standalone file (its explanatory comment block was left out); the function body is unchanged.
node_inventory() {
  local n dev hca cidr want idx f u
  echo "default $(ip -o -4 route show default 2>/dev/null | sed -n 's/.* dev \([^ ]*\).*/\1/p' | head -1)"
  [[ -z "${1:-}" ]] || { ip -o -4 route get "$1" >/dev/null 2>&1 && echo "master ok" || echo "master none"; }
  for n in $(for n in /sys/class/net/*; do
               [[ -d $n/device/infiniband ]] || continue
               u=$(ls "$n/device/infiniband_verbs" 2>/dev/null | head -1); u=${u#uverbs}
               [[ "$u" =~ ^[0-9]+$ ]] || u=9999; echo "$u $n"
             done | sort -n | cut -d' ' -f2); do
    dev=${n##*/}
    [[ -d $n/device/infiniband && "$(cat "$n/operstate" 2>/dev/null)" == up ]] || continue
    hca=$(ls "$n/device/infiniband" | head -1)
    for cidr in $(ip -o -4 addr show dev "$dev" | awk '{print $4}'); do
      # shellcheck disable=SC2046
      want=$(printf '0000:0000:0000:0000:0000:ffff:%02x%02x:%02x%02x' $(tr '.' ' ' <<<"${cidr%/*}"))
      idx=-
      for f in /sys/class/infiniband/"$hca"/ports/1/gids/*; do
        [[ "$(cat "$f" 2>/dev/null)" == "$want" ]] || continue
        [[ "$(cat /sys/class/infiniband/"$hca"/ports/1/gid_attrs/types/"${f##*/}" 2>/dev/null)" == *v2* ]] && { idx=${f##*/}; break; }
      done
      echo "$dev $hca $cidr $idx"
    done
  done
}
