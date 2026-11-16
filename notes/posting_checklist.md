# Posting-Checkliste (für Nils, NICHT automatisch ausführen)

Stand: 16.11.2026, Tag `v1.2-final` (lokal). Es gibt noch kein Remote. Der lokale Branch heißt `master`, nicht `main`.
Alle Befehle in Git Bash im Repo-Wurzelverzeichnis.

## 0. Vorab prüfen

- [ ] `report/summary_en.md` und `review/summary_en.pdf` enthalten deine E-Mail-Adresse und die arXiv-Nummern 2608.12408 und 2611.15533. Mit dem Push werden sie öffentlich. Die arXiv-Nummern sind wie vorgegeben übernommen, ich habe sie nicht nachgeprüft.
- [ ] `git status` ist sauber, `git tag -l` zeigt `v1.2-final`.
- [ ] Optional: in `review/report.pdf` und `review/summary_en.pdf` noch einmal durchblättern.

## 1. GitHub-Repo anlegen und pushen

- [ ] Auf github.com unter `nilsleut` ein neues **öffentliches** Repo `analog-geometry` anlegen, **ohne** README, Lizenz oder .gitignore (das Repo hat schon alles).
- [ ] Remote setzen und pushen (lokales `master` wird auf GitHub zu `main`):
  ```bash
  git remote add origin https://github.com/nilsleut/analog-geometry.git
  git branch -m master main          # optional: lokal ebenfalls in main umbenennen
  git push -u origin main
  git push origin v1.2-final
  ```
  (Ohne Umbenennung: `git push -u origin master:main`.) Die Tags `v1.0-review` und `v1.1-review` nur pushen, wenn du sie öffentlich haben willst.
- [ ] Auf GitHub prüfen: README wird angezeigt, LICENSE (MIT) wird erkannt, `review/` ist vorhanden.

## 2. Repo-Link eintragen

- [ ] `README.md`: Repo-Link ergänzen, z. B. als Zeile unter dem Titel: `Repository: https://github.com/nilsleut/analog-geometry`.
- [ ] `report/summary_en.md`: Platzhalter `[LINK-TO-REPO]` ersetzen.
- [ ] `[LINK-TO-ISSUE]` vorerst stehen lassen (siehe Schritt 4).

## 3. Issue auf IBM/aihwkit eröffnen

- [ ] https://github.com/IBM/aihwkit/issues/new aufrufen. Falls eine Vorlage abgefragt wird, „Question“ bzw. die passendste wählen.
- [ ] **Titel:** die erste Zeile von `review/issue9.md` ohne das führende `# `.
- [ ] **Text:** alles ab der zweiten Zeile von `review/issue9.md`.
- [ ] `figures/fig0_headline.png` per Drag & Drop in den Editor ziehen. GitHub fügt dabei eine Zeile `![...](https://github.com/user-attachments/...)` ein.
  - Damit die Zeile `![...](PLACEHOLDER/fig0_headline.png)` ersetzen.
  - Den HTML-Kommentar `<!-- When posting: ... -->` darüber löschen.
- [ ] Am Ende ergänzen: `Code and full results: https://github.com/nilsleut/analog-geometry (tag v1.2-final).`
- [ ] Vorschau („Preview“) prüfen:
  - Die Tabelle wird gerendert.
  - Der `<details>`-Block lässt sich aufklappen.
  - #702 und #833 sind verlinkt.
  - Das Bild wird angezeigt.
- [ ] Absenden.

## 4. Issue-Link eintragen, neu bauen, pushen

- [ ] `report/summary_en.md`: `[LINK-TO-ISSUE]` durch den Issue-Link ersetzen.
- [ ] Zahlenprüfung und Review-Paket neu bauen:
  ```bash
  MSYS_NO_PATHCONV=1 wsl -d Ubuntu-24.04 --cd /mnt/c/Users/nilsl/Desktop/Projekte_2/analog-geometry -- bash -c 'PYTHONPATH=src ~/.venvs/analog-geometry/bin/python scripts/check_report_numbers.py'
  bash scripts/build_review.sh
  ```
  Erwartet: Report 122 Zahlen, Zusammenfassung 19 Zahlen, jeweils 0 Probleme.
- [ ] Committen und pushen:
  ```bash
  git add README.md report/summary_en.md review/ notes/number_check.md
  git commit -m "Repo- und Issue-Link eingetragen"
  git push
  ```
- [ ] Optional: neuen Tag `v1.2.1` für den Stand mit Links setzen und pushen. `v1.2-final` bleibt dann der Stand, der im Issue genannt ist.
