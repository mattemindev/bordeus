#!/bin/bash
# Ingestion dell'area Sub-ATO E (TeknoService Italia).
#
# Non c'è più nessun elenco di comuni o di URL qui: area, comuni,
# frazioni e calendari sono dichiarati in
# knowledge/sub-ato-e/area.toml, che è anche il posto dove si dichiara
# a quali comuni si applica ciascun calendario.
set -euo pipefail

uv run bordeus-ingest sync --area=sub-ato-e

# Per correggere una data senza ricalcolare gli embedding dell'intera
# area (che su GPU Pascal costa minuti):
#
#   uv run bordeus-ingest sync --area=sub-ato-e --only=calendari
