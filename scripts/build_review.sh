#!/usr/bin/env bash
# Builds review/ (Git Bash on the Windows host): report.pdf from report/report.md (pandoc gfm -> HTML -> headless Chrome),
# fig1-3 as PNG, issue9.md (GitHub-ready draft #9 from notes/aihwkit_issues_draft.md, internal draft numbers removed).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p review
cp figures/fig1_drift_variants.png figures/fig2_noise_levels.png figures/fig3_delta_w.png review/
(cd report && pandoc -f gfm report.md -s --embed-resources --resource-path=. --metadata pagetitle="analog-geometry Report" \
   --metadata lang=de -c ../scripts/review.css -o ../review/_report.html)
CHROME="${CHROME:-/c/Program Files/Google/Chrome/Application/chrome.exe}"
"$CHROME" --headless=new --disable-gpu --no-pdf-header-footer --print-to-pdf="$(cygpath -w "$PWD/review/report.pdf")" \
  "file:///$(cygpath -m "$PWD/review/_report.html")" 2>/dev/null
rm -f review/_report.html
{ echo '# ReRamCMONoiseModel: single-device converter with drift-free reference turns conductance-independent drift into a uniform weight offset. Is this intended?'
  echo; echo '_Question, not a bug report. aihwkit 1.2.0 (= master @ 350ff34), CPU._'; echo
  sed -n '/^\*\*Question\.\*\*/,$p' notes/aihwkit_issues_draft.md \
    | sed 's/ (no constructor argument, see #8)/ (no constructor argument)/; s/ (w_min -> g_min, zero sits mid-range; see #6)/ (w_min -> g_min, zero sits mid-range)/'
} > review/issue9.md
echo "review/ built"
