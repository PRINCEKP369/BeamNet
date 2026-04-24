#!/usr/bin/env bash

# =============================================================================

# procmon.sh — Single-process terminal monitor

# Usage:  ./procmon.sh <PID|process_name> [refresh_interval_seconds]

# Example: ./procmon.sh firefox 2

# =============================================================================

# ── Config ────────────────────────────────────────────────────────────────────

REFRESH=${2:-2}           # default refresh every 2 s
HISTORY_LEN=40            # width of sparkline history bars
MAX_THREADS_SHOWN=10      # max thread rows to display

# ── Colours & styles ──────────────────────────────────────────────────────────

RST=’\033[0m’
BOLD=’\033[1m’
DIM=’\033[2m’
REV=’\033[7m’

BLK=’\033[30m’; RED=’\033[31m’; GRN=’\033[32m’; YLW=’\033[33m’
BLU=’\033[34m’; MAG=’\033[35m’; CYN=’\033[36m’; WHT=’\033[37m’

BBLK=’\033[90m’; BRED=’\033[91m’; BGRN=’\033[92m’; BYLW=’\033[93m’
BBLU=’\033[94m’; BMAG=’\033[95m’; BCYN=’\033[96m’; BWHT=’\033[97m’

BG_HDR=’\033[48;5;235m’   # dark grey header background
BG_SEC=’\033[48;5;233m’   # section background

# ── Helpers ───────────────────────────────────────────────────────────────────

resolve_pid() {
local arg=”$1”
if [[ “$arg” =~ ^[0-9]+$ ]]; then
echo “$arg”
else
pgrep -x “$arg” | head -1
fi
}

human_bytes() {
local b=$1
if   (( b >= 1073741824 )); then printf “%.1f GiB” “$(echo “scale=1; $b/1073741824” | bc)”
elif (( b >= 1048576    )); then printf “%.1f MiB” “$(echo “scale=1; $b/1048576”    | bc)”
elif (( b >= 1024       )); then printf “%.1f KiB” “$(echo “scale=1; $b/1024”       | bc)”
else                              printf “%d B”    “$b”
fi
}

# Horizontal bar  fill_bar <value_0-100> <width> <fill_char> <fill_color>

fill_bar() {
local val=$1 width=$2 ch=”${3:-█}” color=”${4:-$BGRN}”
local filled=$(( val * width / 100 ))
(( filled > width )) && filled=$width
local empty=$(( width - filled ))
printf “${color}”
printf “%0.s${ch}” $(seq 1 $filled) 2>/dev/null || printf ‘%*s’ “$filled” ‘’ | tr ’ ’ “$ch”
printf “${BBLK}”
printf ‘%*s’ “$empty” ‘’ | tr ’ ’ ‘░’
printf “${RST}”
}

# Colour a percentage value

pct_color() {
local v=${1%.*}   # integer part
if   (( v >= 80 )); then printf “${BRED}%s${RST}” “$1”
elif (( v >= 50 )); then printf “${BYLW}%s${RST}” “$1”
else                     printf “${BGRN}%s${RST}” “$1”
fi
}

# Append to circular history array (global name passed as $1)

push_history() {
local -n _arr=$1
_arr+=(”$2”)
while (( ${#_arr[@]} > HISTORY_LEN )); do
_arr=(”${_arr[@]:1}”)
done
}

# Render sparkline from history array values (0-100)

sparkline() {
local -n _h=$1
local blocks=(’ ’ ‘▁’ ‘▂’ ‘▃’ ‘▄’ ‘▅’ ‘▆’ ‘▇’ ‘█’)
local out=””
for v in “${_h[@]}”; do
local idx=$(( v * 8 / 100 ))
(( idx > 8 )) && idx=8
out+=”${blocks[$idx]}”
done
printf “%s” “$out”
}

# ── Startup check ─────────────────────────────────────────────────────────────

if [[ -z “$1” ]]; then
echo -e “${BRED}Usage:${RST} $0 <PID|process_name> [refresh_seconds]”
exit 1
fi

TARGET_PID=$(resolve_pid “$1”)
if [[ -z “$TARGET_PID” ]]; then
echo -e “${BRED}Error:${RST} process ‘$1’ not found.”
exit 1
fi

# ── State ─────────────────────────────────────────────────────────────────────

declare -a cpu_history=()
declare -a mem_history=()
declare -a rss_history=()
declare -a io_r_history=()
declare -a io_w_history=()

prev_io_r=0; prev_io_w=0
sample=0
start_ts=$(date +%s)

# ── Cleanup on exit ───────────────────────────────────────────────────────────

cleanup() {
tput rmcup 2>/dev/null || true
tput cnorm 2>/dev/null || true
echo -e “${RST}”
}
trap cleanup EXIT INT TERM

tput smcup 2>/dev/null || true
tput civis 2>/dev/null || true   # hide cursor

# ── Main loop ─────────────────────────────────────────────────────────────────

while true; do

```
# ── Check process still exists ────────────────────────────────────────────
if [[ ! -d "/proc/$TARGET_PID" ]]; then
    tput cup 0 0
    echo -e "\n  ${BRED}Process $TARGET_PID has exited.${RST}\n"
    sleep 2; exit 0
fi

# ── Gather /proc data ─────────────────────────────────────────────────────
# stat fields
read -r stat_line < "/proc/$TARGET_PID/stat" 2>/dev/null || { sleep "$REFRESH"; continue; }
IFS=' ' read -ra sf <<< "$stat_line"
proc_name_raw="${sf[1]}"                          # (name)
proc_name="${proc_name_raw//[()]/}"
proc_state="${sf[2]}"
proc_ppid="${sf[3]}"
proc_nice="${sf[18]}"
proc_num_threads="${sf[19]}"
proc_starttime="${sf[21]}"
utime="${sf[13]}"; stime="${sf[14]}"
total_jiffies=$(( utime + stime ))

# uptime / Hz
read -r sys_uptime _ < /proc/uptime
CLK_TCK=$(getconf CLK_TCK 2>/dev/null || echo 100)
proc_uptime_s=$(echo "scale=2; $sys_uptime - $proc_starttime / $CLK_TCK" | bc 2>/dev/null || echo 0)
# running time as HH:MM:SS
pu_int=${proc_uptime_s%.*}
pu_h=$(( pu_int / 3600 ))
pu_m=$(( (pu_int % 3600) / 60 ))
pu_s=$(( pu_int % 60 ))
proc_uptime_fmt=$(printf "%02d:%02d:%02d" $pu_h $pu_m $pu_s)

# status fields (VmRSS, VmVirt, VmSwap, uid)
vmrss_kb=0; vmvirt_kb=0; vmswap_kb=0; proc_uid=0
while IFS=': ' read -r key val unit; do
    case "$key" in
        VmRSS)  vmrss_kb=${val//[^0-9]/}  ;;
        VmSize) vmvirt_kb=${val//[^0-9]/} ;;
        VmSwap) vmswap_kb=${val//[^0-9]/} ;;
        Uid)    proc_uid=$(awk '{print $1}' <<< "$val") ;;
    esac
done < "/proc/$TARGET_PID/status" 2>/dev/null

proc_user=$(getent passwd "$proc_uid" 2>/dev/null | cut -d: -f1 || echo "$proc_uid")

# exe / cmdline
proc_exe=$(readlink -f "/proc/$TARGET_PID/exe" 2>/dev/null || echo "(unknown)")
proc_cmd=$(tr '\0' ' ' < "/proc/$TARGET_PID/cmdline" 2>/dev/null | cut -c1-80)

# cwd
proc_cwd=$(readlink "/proc/$TARGET_PID/cwd" 2>/dev/null || echo "(unknown)")

# open file descriptors
proc_fds=$(ls /proc/$TARGET_PID/fd 2>/dev/null | wc -l)

# CPU %  (via ps for simplicity + accuracy over short windows)
cpu_pct=$(ps -p "$TARGET_PID" -o %cpu= 2>/dev/null | tr -d ' ' || echo "0")
cpu_int=${cpu_pct%.*}; (( cpu_int > 100 )) && cpu_int=100

# Memory %
mem_pct=$(ps -p "$TARGET_PID" -o %mem= 2>/dev/null | tr -d ' ' || echo "0")
mem_int=${mem_pct%.*}

# Total system RAM
total_mem_kb=$(awk '/MemTotal/{print $2}' /proc/meminfo)
total_mem_str=$(human_bytes $(( total_mem_kb * 1024 )))

# I/O counters
io_r=0; io_w=0
if [[ -r "/proc/$TARGET_PID/io" ]]; then
    while IFS=': ' read -r key val; do
        case "$key" in
            read_bytes)  io_r=$val ;;
            write_bytes) io_w=$val ;;
        esac
    done < "/proc/$TARGET_PID/io" 2>/dev/null
fi
io_r_delta=$(( io_r - prev_io_r ))
io_w_delta=$(( io_w - prev_io_w ))
(( io_r_delta < 0 )) && io_r_delta=0
(( io_w_delta < 0 )) && io_w_delta=0
prev_io_r=$io_r; prev_io_w=$io_w
io_r_rate=$(human_bytes $io_r_delta)/s
io_w_rate=$(human_bytes $io_w_delta)/s
io_r_total=$(human_bytes $io_r)
io_w_total=$(human_bytes $io_w)

# Clamp I/O for history bar (cap at 10 MiB/s = 100%)
io_r_pct=$(( io_r_delta * 100 / 10485760 ))
io_w_pct=$(( io_w_delta * 100 / 10485760 ))
(( io_r_pct > 100 )) && io_r_pct=100
(( io_w_pct > 100 )) && io_w_pct=100

# Number of CPUs
ncpus=$(nproc 2>/dev/null || grep -c ^processor /proc/cpuinfo)

# Push history
push_history cpu_history "$cpu_int"
push_history mem_history "$mem_int"
push_history rss_history "$(( vmrss_kb / 1024 ))"   # MiB values, not %
push_history io_r_history "$io_r_pct"
push_history io_w_history "$io_w_pct"

(( sample++ ))
wall_elapsed=$(( $(date +%s) - start_ts ))

# State colour
case "$proc_state" in
    R) state_color="${BGRN}"; state_label="Running"  ;;
    S) state_color="${BCYN}"; state_label="Sleeping" ;;
    D) state_color="${BRED}"; state_label="Disk Wait" ;;
    Z) state_color="${BRED}"; state_label="Zombie"   ;;
    T) state_color="${BYLW}"; state_label="Stopped"  ;;
    *) state_color="${BWHT}"; state_label="$proc_state" ;;
esac

# ── Draw ──────────────────────────────────────────────────────────────────
COLS=$(tput cols 2>/dev/null || echo 80)
ROWS=$(tput lines 2>/dev/null || echo 24)
BAR_W=$(( COLS - 22 ))
(( BAR_W < 10 )) && BAR_W=10

tput cup 0 0
# clear screen efficiently
tput ed 2>/dev/null || clear

# ── HEADER ────────────────────────────────────────────────────────────────
header_text="  ◉  PROCMON  │  PID: $TARGET_PID  │  $proc_name  │  $(date '+%Y-%m-%d %H:%M:%S')  "
printf "${BG_HDR}${BOLD}${BCYN}%-${COLS}s${RST}\n" "$header_text"
printf "${BG_HDR}${DIM}${BBLK}  Monitor elapsed: ${wall_elapsed}s  │  Samples: ${sample}  │  Refresh: ${REFRESH}s%-$((COLS-55))s${RST}\n" ""

# ── IDENTITY BLOCK ────────────────────────────────────────────────────────
printf "\n"
printf "  ${BOLD}${BCYN}PROCESS IDENTITY${RST}\n"
printf "  ${BBLK}────────────────────────────────────────────────────────────────────${RST}\n"
printf "  ${DIM}Name    ${RST}${BWHT}%-20s${RST}  ${DIM}State   ${RST}${state_color}${BOLD}%-12s${RST}  ${DIM}PID     ${RST}${BYLW}%s${RST}\n" \
    "$proc_name" "$state_label" "$TARGET_PID"
printf "  ${DIM}User    ${RST}${BWHT}%-20s${RST}  ${DIM}PPID    ${RST}${BWHT}%-12s${RST}  ${DIM}Threads ${RST}${BWHT}%s${RST}\n" \
    "$proc_user" "$proc_ppid" "$proc_num_threads"
printf "  ${DIM}Nice    ${RST}${BWHT}%-20s${RST}  ${DIM}Up time ${RST}${BWHT}%-12s${RST}  ${DIM}FDs     ${RST}${BWHT}%s${RST}\n" \
    "$proc_nice" "$proc_uptime_fmt" "$proc_fds"
printf "  ${DIM}Exe     ${RST}${BWHT}%s${RST}\n" "$proc_exe"
printf "  ${DIM}CWD     ${RST}${BWHT}%s${RST}\n" "$proc_cwd"
printf "  ${DIM}Command ${RST}${BWHT}%s${RST}\n" "$proc_cmd"

# ── CPU ───────────────────────────────────────────────────────────────────
printf "\n"
printf "  ${BOLD}${BGRN}CPU USAGE${RST}  ${DIM}(${ncpus} logical cores)${RST}\n"
printf "  ${BBLK}────────────────────────────────────────────────────────────────────${RST}\n"
printf "  %-8s " "CPU %"
fill_bar "$cpu_int" "$BAR_W"
printf "  "
pct_color "${cpu_pct}%"
printf "\n"
printf "  ${DIM}History ${RST}${BCYN}%s${RST}\n" "$(sparkline cpu_history)"

# ── MEMORY ────────────────────────────────────────────────────────────────
rss_str=$(human_bytes $(( vmrss_kb * 1024 )))
virt_str=$(human_bytes $(( vmvirt_kb * 1024 )))
swap_str=$(human_bytes $(( vmswap_kb * 1024 )))
printf "\n"
printf "  ${BOLD}${BMAG}MEMORY USAGE${RST}  ${DIM}(system total: ${total_mem_str})${RST}\n"
printf "  ${BBLK}────────────────────────────────────────────────────────────────────${RST}\n"
printf "  %-8s " "RSS %"
fill_bar "$mem_int" "$BAR_W" "█" "$BMAG"
printf "  "
pct_color "${mem_pct}%"
printf "\n"
printf "  ${DIM}RSS     ${RST}${BWHT}%-14s${RST}  ${DIM}Virtual ${RST}${BWHT}%-14s${RST}  ${DIM}Swap    ${RST}${BWHT}%s${RST}\n" \
    "$rss_str" "$virt_str" "$swap_str"
printf "  ${DIM}History ${RST}${BMAG}%s${RST}\n" "$(sparkline mem_history)"

# ── I/O ───────────────────────────────────────────────────────────────────
printf "\n"
printf "  ${BOLD}${BYLW}DISK I/O${RST}\n"
printf "  ${BBLK}────────────────────────────────────────────────────────────────────${RST}\n"
printf "  %-8s " "Read/s"
fill_bar "$io_r_pct" "$BAR_W" "█" "$BGRN"
printf "  ${BGRN}%s${RST}\n" "$io_r_rate"
printf "  %-8s " "Write/s"
fill_bar "$io_w_pct" "$BAR_W" "█" "$BYLW"
printf "  ${BYLW}%s${RST}\n" "$io_w_rate"
printf "  ${DIM}Read total ${RST}${BWHT}%-14s${RST}  ${DIM}Write total ${RST}${BWHT}%s${RST}\n" \
    "$io_r_total" "$io_w_total"
printf "  ${DIM}Read  hist ${RST}${BGRN}%s${RST}\n" "$(sparkline io_r_history)"
printf "  ${DIM}Write hist ${RST}${BYLW}%s${RST}\n" "$(sparkline io_w_history)"

# ── THREADS ───────────────────────────────────────────────────────────────
printf "\n"
printf "  ${BOLD}${BBLU}THREADS${RST}  ${DIM}(showing up to ${MAX_THREADS_SHOWN} of ${proc_num_threads})${RST}\n"
printf "  ${BBLK}────────────────────────────────────────────────────────────────────${RST}\n"
printf "  ${DIM}%-8s  %-12s  %-10s  %-10s  %s${RST}\n" "TID" "State" "User(j)" "Sys(j)" "Name"
count=0
for tid_dir in /proc/$TARGET_PID/task/*/; do
    tid=$(basename "$tid_dir")
    [[ ! -f "${tid_dir}stat" ]] && continue
    IFS=' ' read -ra tf < "${tid_dir}stat"
    t_name="${tf[1]//[()]/}"
    t_state="${tf[2]}"
    t_utime="${tf[13]}"
    t_stime="${tf[14]}"
    case "$t_state" in
        R) ts_col="${BGRN}" ;;  S) ts_col="${BCYN}" ;;
        D) ts_col="${BRED}" ;;  *) ts_col="${BWHT}" ;;
    esac
    printf "  ${BWHT}%-8s${RST}  ${ts_col}%-12s${RST}  %-10s  %-10s  ${DIM}%s${RST}\n" \
        "$tid" "$t_state" "$t_utime" "$t_stime" "$t_name"
    (( ++count >= MAX_THREADS_SHOWN )) && break
done
(( proc_num_threads > MAX_THREADS_SHOWN )) && \
    printf "  ${BBLK}  … and %d more threads${RST}\n" $(( proc_num_threads - MAX_THREADS_SHOWN ))

# ── OPEN FILES (top 5) ────────────────────────────────────────────────────
printf "\n"
printf "  ${BOLD}${BCYN}OPEN FILE DESCRIPTORS${RST}  ${DIM}(top 5 of ${proc_fds})${RST}\n"
printf "  ${BBLK}────────────────────────────────────────────────────────────────────${RST}\n"
fd_count=0
for fd in /proc/$TARGET_PID/fd/*; do
    target=$(readlink "$fd" 2>/dev/null || echo "(unresolvable)")
    fd_num=$(basename "$fd")
    printf "  ${BBLK}fd%-4s${RST}  ${DIM}%s${RST}\n" "$fd_num" "$target"
    (( ++fd_count >= 5 )) && break
done

# ── FOOTER ────────────────────────────────────────────────────────────────
printf "\n"
printf "  ${BBLK}Press ${RST}${BWHT}Ctrl+C${RST}${BBLK} to exit  │  Refreshing every ${REFRESH}s${RST}\n"

sleep "$REFRESH"
```

done
