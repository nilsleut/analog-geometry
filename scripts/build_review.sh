#!/usr/bin/env bash
# Builds review/ (Git Bash on the Windows host; needs pandoc and Chrome):
#   report.pdf, summary_en.pdf  from report/*.md (pandoc gfm -> self-contained HTML -> headless Chrome)
#   fig0_headline.png, fig1-3   copied from figures/
#   issue9.md                   GitHub-ready draft #9 from notes/aihwkit_issues_draft.md (internal draft numbers removed)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p review
cp figures/fig0_headline.png figures/fig1_drift_variants.png figures/fig2_noise_levels.png figures/fig3_delta_w.png review/
CHROME="${CHROME:-/c/Program Files/Google/Chrome/Application/chrome.exe}"

render() {  # $1 = markdown in report/, $2 = output pdf name, $3 = lang, $4 = optional extra css
  (cd report && pandoc -f gfm "$1" -s --embed-resources --resource-path=. --metadata pagetitle="analog-geometry" \
     --metadata lang="$3" -c ../scripts/review.css ${4:+-c ../scripts/$4} -o ../review/_tmp.html)
  "$CHROME" --headless=new --disable-gpu --no-pdf-header-footer --print-to-pdf="$(cygpath -w "$PWD/review/$2")" \
    "file:///$(cygpath -m "$PWD/review/_tmp.html")" 2>/dev/null
  rm -f review/_tmp.html
}
render report.md report.pdf de
render summary_en.md summary_en.pdf en review_summary.css

{ echo '# ReRamCMONoiseModel: single-device converter with drift-free reference turns conductance-independent drift into a uniform weight offset. Is this intended?'
  echo; echo '_Question, not a bug report. aihwkit 1.2.0 (= master @ 350ff34), CPU._'; echo
  sed -n '/^\*\*Question\.\*\*/,$p' notes/aihwkit_issues_draft.md \
    | sed 's/ (no constructor argument, see #8)/ (no constructor argument)/; s/ (w_min -> g_min, zero sits mid-range; see #6)/ (w_min -> g_min, zero sits mid-range)/'
} > review/issue9.md
echo "review/ built"
