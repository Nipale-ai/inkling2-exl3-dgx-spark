#!/usr/bin/env bash
# rdma_check.sh LINKS_FILE: ib_write_bw (5 s, 1 MiB) over every cable of the ring, one line per cable in LINKS_FILE:
#   "server_host server_ip server_dev client_host client_dev"   (ssh names, IP of the server cable end, RoCE device names, GID index 3)
# Pass: every link >= MIN_GBPS (default 100). Needs idle GPUs/NICs and perftest (ib_write_bw) on all hosts. Run from a host that can ssh to all of them.
MIN=${MIN_GBPS:-100}
ok=1
# the host this script runs on is addressed locally (a box usually has no ssh key to itself)
rx() { if [ "$1" = "$(hostname)" ] || [ "$1" = "$(hostname -s)" ]; then bash -c "$2"; else ssh -n "$1" "$2"; fi; }
while read -r SH SIP SD CH CD; do
  [ -z "$SH" ] && continue
  rx $SH "pkill -u \$USER -x ib_write_bw; nohup timeout 20 ib_write_bw -d $SD -x 3 -s 1048576 -D 5 -F --report_gbits -p 18515 >/tmp/ibw-srv.log 2>&1 &"
  sleep 2
  out=$(rx $CH "timeout 30 ib_write_bw -d $CD -x 3 -s 1048576 -D 5 -F --report_gbits -p 18515 $SIP 2>&1 | grep -E '^ *1048576'")
  bw=$(echo "$out" | awk '{print $4}' | head -1)
  echo "$SH:$SD <-> $CH:$CD ($SIP): ${bw:-none} Gb/s (min $MIN)"
  awk -v b="${bw:-0}" -v m="$MIN" 'BEGIN{exit !(b+0>=m+0)}' || ok=0
done < "${1:?usage: rdma_check.sh LINKS_FILE}"
[ $ok = 1 ] && echo "RDMA OK" || echo "RDMA FAIL"
