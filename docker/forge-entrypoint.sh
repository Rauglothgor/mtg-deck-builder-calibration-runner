#!/usr/bin/env bash
set -euo pipefail

FORGE_HOME="${FORGE_HOME:-/opt/forge}"
FORGE_USER_DIR="${FORGE_USER_DIR:-/root/.forge}"
COMMANDER_DIR="${FORGE_USER_DIR}/decks/commander"
FORGE_BIN="${FORGE_HOME}/forge.sh"
DEFAULT_DECK_A="atraxa.dck"
DEFAULT_DECK_B="alela.dck"

print_help() {
  cat <<'EOT'
Usage:
  forge-entrypoint.sh --help
  forge-entrypoint.sh list-decks
  forge-entrypoint.sh smoke-test
  forge-entrypoint.sh sim -d <deck1> <deck2> [-n <games>] [-f <format>]
  forge-entrypoint.sh <deck1.dck> <deck2.dck> [games]
EOT
}

list_decks() {
  find "${COMMANDER_DIR}" -maxdepth 1 -type f -name '*.dck' -printf '%f\n' | sort
}

resolve_deck() {
  local deck="$1"

  if [[ -f "$deck" ]]; then
    printf '%s\n' "$deck"
    return 0
  fi

  if [[ -f "${COMMANDER_DIR}/${deck}" ]]; then
    printf '%s\n' "${COMMANDER_DIR}/${deck}"
    return 0
  fi

  echo "Deck not found: ${deck}" >&2
  exit 2
}

run_sim() {
  local deck_a
  local deck_b
  local games="$3"

  deck_a="$(basename "$(resolve_deck "$1")")"
  deck_b="$(basename "$(resolve_deck "$2")")"

  exec "${FORGE_BIN}" sim -D "${COMMANDER_DIR}" -d "${deck_a}" "${deck_b}" -n "${games}" -f commander
}

case "${1:---help}" in
  --help|-h|help)
    print_help
    ;;
  list-decks)
    list_decks
    ;;
  smoke-test)
    run_sim "${DEFAULT_DECK_A}" "${DEFAULT_DECK_B}" 1
    ;;
  sim)
    shift
    games="1"
    format="commander"
    decks=()

    while [[ "$#" -gt 0 ]]; do
      case "$1" in
        -d)
          shift
          while [[ "$#" -gt 0 ]]; do
            case "$1" in
              -n|-f|-q)
                break
                ;;
              *)
                decks+=("$1")
                shift
                ;;
            esac
          done
          ;;
        -n)
          [[ "$#" -ge 2 ]] || { echo "Missing value for -n" >&2; exit 2; }
          games="$2"
          shift 2
          ;;
        -f)
          [[ "$#" -ge 2 ]] || { echo "Missing value for -f" >&2; exit 2; }
          format="$2"
          shift 2
          ;;
        -q)
          shift 1
          ;;
        *)
          echo "Unsupported argument: $1" >&2
          exit 2
          ;;
      esac
    done

    [[ "${#decks[@]}" -eq 2 ]] || {
      echo "sim requires exactly two deck names after -d" >&2
      exit 2
    }

    run_sim "${decks[0]}" "${decks[1]}" "$games"
    ;;
  *)
    if [[ "$#" -lt 2 || "$#" -gt 3 ]]; then
      echo "Usage: forge-entrypoint.sh <deck1.dck> <deck2.dck> [games]" >&2
      exit 2
    fi
    run_sim "$1" "$2" "${3:-1}"
    ;;
esac
