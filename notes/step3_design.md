# Schritt 3: Designentscheidungen

Grundsatz: Digital trainierte Gewichte werden per `convert_to_analog` übernommen. Es gibt nur Inferenz, kein hardware-bewusstes Training.
Ablauf pro Konfiguration: program_analog_weights() -> drift_analog_weights(t) -> Aktivierungen auf dem 1000er-Set -> CKA/RSA gegen das digitale Netz desselben Seeds.

## Verfügbare ReRAM-Modelle in aihwkit 1.2.0
| | ReRamWan2022NoiseModel | ReRamCMONoiseModel |
|---|---|---|
| Quelle | Wan et al., Nature 2022 | IBM Zürich, CMO/HfOx (Falcone et al., in review) |
| Programmierrauschen getrennt skalierbar | nein (ein `noise_scale` für alles) | ja (`prog_noise_scale`) |
| Drift-Zeit | nur 1 s, 1 d, 2 d | kontinuierlich (log t) |
| Drift-Semantik | Neuziehen aus g_target, unabhängig vom programmierten Wert | Mittelwertverschiebung + Relaxation + Read-Noise auf g_prog |
| Mapping | Single Pair (differentiell), g_max 25 µS | Single Device, g 9–88.2 µS |

## Quantisierung auf k Stufen
Eigene Erweiterung: g_target wird vor dem Programmierrauschen auf k äquidistante Stufen in [g_min, g_max] gerundet. Kein aihwkit-Feature, wird im Report so benannt.

## Peripherie (ADC/DAC, Read-Noise im Forward)
Default-InferenceRPUConfig enthält zusätzlich w_noise, out_noise, inp_res/out_res und Bound/Noise-Management.
Optionen: (i) Peripherie ideal setzen, um nur Geräteeffekte zu isolieren; (ii) Defaults konstant lassen.

## ENTSCHEIDUNGEN (Nutzer, 2026-11-14)
1. Hauptmodell ReRamCMONoiseModel; Wan2022 bei 1 s / 1 d / 2 d als Gegenprobe (nur Anhang bzw. Zusatzzeile, nicht im Hauptbild).
2. Akzeptanzbereich 2 % explizit (Umgehung im eigenen Code, siehe ENVIRONMENT.md). Programmier-Rauschskalierung 0,1,2,4,8,16,32,64.
   Eskalation: Ist der CIFAR-Accuracy-Verlust bei 64 nicht größer als 2 pp, um 128 und 256 erweitern (`--extra-prog 128 256`) und melden.
3. Peripherie ideal (`forward.is_perfect=True`). Zusatzpunkt mit aihwkit-Defaults bei Skalierung 1, Drift 1 d.
4. Kein HWA-Training (mögliche Erweiterung nach dem 14.12.).
5. Pro Trainings-Seed und Einstellung 3 unabhängige Programmier-Durchgänge (noise_seed = 1000*seed + rep). Varianz getrennt ausweisen: zwischen Seeds / zwischen Durchgängen.
6. Ein-Faktor-Sweeps: Drift (CMO) 1 s, 1 h, 1 d, 1 Woche, 1 Jahr bei Skalierung 1. Stufen 8/16/32/64/inf ohne Rauschen und Drift. Rauschskalierung bei Drift 1 s.
   Hinweis: Auch bei Skalierung 0 und t = 1 s wirken CMO-Relaxation und Read-Noise; nur der Punkt "ideal" ist rauschfrei.
7. Referenzband: CKA/RSA zwischen allen 10 Paaren digitaler Seeds. Nur Abweichungen außerhalb des Bands zählen als Befund.
8. Zuerst MNIST als Probelauf, dann melden, erst danach CIFAR.
9. Bugs vor jedem Entwurf gegen den aktuellen Upstream-Default-Branch prüfen; nichts posten.

## VORAB-FESTLEGUNG: Befundregel (festgelegt 2026-11-15, vor dem Start des CIFAR-Sweeps; danach nicht mehr ändern)
Eine Einstellung gilt als Befund „Accuracy stabil, Geometrie weicht ab“ genau dann, wenn BEIDE Bedingungen erfüllt sind:
1. **Accuracy stabil:** Verlust ≤ 1 pp gegenüber der digitalen Baseline desselben Seeds (Testset, 10 000 Bilder), gemittelt über die
   3 Programmier-Durchgänge und dann über die 5 Seeds (Spalte `acc_drop_pp_mean` in `results/sweep_summary_<ds>.csv`).
2. **Geometrie außerhalb des Bands:** In mindestens einer Schicht liegt der Mittelwert über 5 Seeds × 3 Durchgänge von CKA ODER RSA
   unter dem Minimum der 10 digitalen Seed-Paare dieser Schicht (Spalten `cka_outside_band` / `rsa_outside_band`).
Benennung: Gilt Bedingung 2 nur für CKA oder nur für RSA, wird der Befund so benannt („nur CKA“ / „nur RSA“) und nicht als
allgemeine „Geometrie-Abweichung“ formuliert.
Versatz-Kontrollen (Interpretation, nicht Teil der Regel): CKA mit pro Stimulus zentrierten Aktivierungen und Drift ohne
Mittelwertverschiebung zeigen, ob eine CKA-Abweichung ein Versatz ist oder eine Umordnung.
Status: Die MNIST-Ergebnisse lagen bei der Festlegung bereits vor (Probelauf). Die Regel ist erst für CIFAR echt vorab festgelegt;
für MNIST wird sie unverändert angewendet, im Report aber als nachträglich gekennzeichnet.

## Drift-Kompensation (geprüft 2026-11-15)
- `InferenceRPUConfig()` setzt per Default `drift_compensation = GlobalDriftCompensation()`. Sie ist in ALLEN Läufen aktiv
  (auch bei `forward.is_perfect=True`; sie wird in `post_forward` im Eval-Modus angewendet).
- Mechanik: Nach dem Programmieren Referenz ref = mean|Ausgabe bei Einheitsmatrix| (≈ mean|W_prog|) pro Tile; nach der Drift
  alpha = ref / mean|W_drift|; jede Tile-Ausgabe wird mit dem Skalar alpha multipliziert.
- Warum trotzdem ein Versatz bleibt: Der CMO-Drift-Mittelwert verschiebt alle Leitwerte gleich. `SingleDeviceConductanceConverter`
  bildet linear ab, also entsteht ein gleichförmiger Gewichtsversatz Δw und damit ein additiver Term Δw·Σx_i pro Stimulus.
  Ein skalarer Faktor kann einen additiven Term nicht entfernen.
- Empirisch (scripts/check_drift_compensation.py -> results/check_drift_compensation.csv; MNIST Seed 0, ein Durchgang):
  alpha ∈ [0.92, 0.99]. CKA mit/ohne Kompensation identisch auf 3 Stellen (CKA ist skaleninvariant). Ohne Drift-Mittelwert:
  CKA ≥ 0.998 bei 1 d und 1 J, kein Accuracy-Verlust. Gleiche Streuung von Δw, mittleres Δw ≈ 0 statt -0.008 / -0.012.
- Nach der Nutzerregel (Kompensation ist an) kein Zusatzpunkt mit Kompensation; stattdessen diese Erklärung.

## Reproduzierbarkeit (geprüft 2026-11-15, MNIST-Neulauf gegen v1)
Alle Einstellungen mit idealer Peripherie sind bitgenau reproduziert (990 von 990 gemeinsamen Zeilen, Differenz 0).
Ausnahme `periphery_default` (45 Zeilen): Das Rauschen im Forward-Pass kommt aus dem C++-RNG von aihwkit, den torch.manual_seed nicht erfasst.
Max |ΔCKA| 1.6e-3, |ΔRSA| 2.0e-3, |ΔAcc| 0.21 pp. Wird im Report als Grenze genannt (siehe issues_draft #7).

## Nachtrag 2026-11-15: Zeitpunkt des ersten Commits
Vorab-Regel (notes/step3_design.md) eingetragen 15.11.2026 vor CIFAR-Start; dieser Commit enthält bereits den CIFAR-Teillauf Rauschskalierung 0/4/16/64.
Der erste Git-Commit des Repos folgt also zeitlich NACH der Regel und NACH dem CIFAR-Teillauf (Rauschskalierung 0/4/16/64, 5 Seeds × 3 Durchgänge,
plus Seed-0-Teilpunkte 1/2/8 und „ideal“). Er belegt die Regel deshalb nicht als vor diesen Läufen festgelegt. Der Regeltext ist seit seiner Eintragung unverändert.
Für alle weiteren CIFAR-Sweeps ist der Commit ein Beleg dafür, dass die Regel vorher feststand.

## Nachtrag 2026-11-15: Grenzfall-Regel (Nutzerentscheidung)
Fall: MNIST, Drift 1 Tag, act1, CKA mit Zentrierung pro Stimulus = 0.9552, Bandminimum = 0.9554 (Abstand 0.0002).
Entscheidung: Dieser Fall gilt als UNENTSCHIEDEN und zählt NICHT als außerhalb des Bands.
Begründung (Nutzer): Der Abstand ist kleiner als die gemessene Nicht-Reproduzierbarkeit von 0.0016 (max |ΔCKA| zwischen zwei identischen Läufen).
Anmerkung: Die 0.0016 stammen vom Punkt `periphery_default` (C++-RNG). Läufe mit idealer Peripherie, also auch dieser Drift-Punkt,
sind bitgenau reproduzierbar. Die Spanne aus Seeds und Durchgängen (cka_stimc_std_total) ist aber ebenfalls größer als 0.0002.
Geltungsbereich: Die Nutzerregel nennt nur diesen einen Fall. Sie berührt nur die Versatz-Kontrolle 2b, nicht die Vorab-Befundregel (die auf cka/rsa beruht).
Weitere Grenzfälle (Abstand < 0.0016 zum Bandminimum) werden einzeln gemeldet und nicht automatisch entschieden.

## Punkt 3: ReLU-Hypothese, MNIST (2026-11-15)
Daten: results/preact_mnist.csv, results/digital_seed_reference_preact_mnist.csv -> results/preact_summary_mnist.csv
(Drift-Reihe + Kontrolle 2a neu gerechnet mit Prä-Aktivierungen, 105 Läufe. Die Accuracies sind identisch zu analog_sweep_mnist.csv,
105/105 geprüft. Statt nur 1 d / 1 Wo ist die ganze Drift-Reihe gerechnet, weil sie auf MNIST billig ist.)
Kriterium (Nutzer): CKA mit Zentrierung pro Stimulus auf den Prä-Aktivierungen. Im Band vor der ReLU und außerhalb danach -> Lesart gestützt.
Schon vorher außerhalb -> Lesart falsch.
- Schicht 1 (fc1 -> act1): vor der ReLU cka_stimc 0.9997-0.9999 bei allen Drift-Zeiten (Band-Min 0.9820), also IM Band.
  Danach (act1): 1 d 0.9552 = UNENTSCHIEDEN (Grenzfall-Regel), 1 Wo 0.9410 und 1 J 0.9097 AUSSERHALB. -> Lesart für Schicht 1 GESTÜTZT (1 Wo, 1 J).
- Schicht 2 (fc2 -> act2): vor der ReLU cka_stimc 1 d 0.9338, 1 Wo 0.9065, 1 J 0.8338 (Band-Min 0.9513), also schon AUSSERHALB.
  -> Lesart für Schicht 2 FALSCH.
  Ungeprüfte Vermutung: Schicht 2 erbt die nicht-subtraktive Veränderung aus act1. Ihr eigener gleichförmiger Versatz Δw·Σ act1 würde durch die Zentrierung entfernt.
- Kontrolle 2a (ohne Drift-Mittelwert): vor und nach der ReLU alles im Band (>= 0.998).
Formulierung für den Report: nur schichtweise, nicht als allgemeine Aussage.

## Nachtrag 2026-11-15: Begründung der Grenzfall-Regel korrigiert (die Regel selbst bleibt unverändert)
Korrektur: Die 0.0016 (Nicht-Reproduzierbarkeit) gelten NUR für den Punkt `periphery_default`. Die Drift-Läufe sind bitgenau reproduzierbar,
die 0.0016 begründen den Grenzfall also nicht.
Begründung für die Drift-Läufe (Nutzer): Die Bandgrenze ist selbst unsicher. Das Minimum stammt aus nur 10 Seed-Paaren (5 Seeds).
Bootstrap über die 5 Seeds (results/band_bootstrap_mnist.csv; B = 10000, Seed 20261116; je Replikat 5 Seeds mit Zurücklegen,
Minimum über Paare verschiedener Seeds):
- act1, cka_stimc: beobachtetes Minimum 0.9554, 95%-Bootstrap-Intervall [0.9554, 0.9632], Median 0.9580.
Einschränkung (wichtig): Der Bootstrap eines Minimums kann nie unter das beobachtete Minimum fallen (Resampling erzeugt keine neuen,
kleineren Paarwerte). Das Intervall zeigt deshalb nur Unsicherheit nach OBEN. Danach läge der Grenzfall 0.9552 eher deutlicher außerhalb.
Die Unsicherheit nach unten (mehr Seeds -> mehr Paare -> tendenziell kleineres Minimum) erfasst diese Methode prinzipiell nicht.
Ergänzend, deskriptiv: Die 10 Paarwerte act1/cka_stimc haben Mittel 0.9609, sd 0.0030. 0.9552 liegt bei Mittel - 1.9 sd,
also im Bereich, den man bei mehr Seed-Paaren unter Normalannahme erwarten würde. Das stützt „unentschieden“, ist aber kein Test.
Fazit: Die Regel bleibt (unentschieden, zählt nicht als außerhalb). Der Bootstrap stützt die Begründung NICHT direkt;
gestützt wird sie nur durch das deskriptive Argument oben.

## Punkt 1 (2026-11-15): Schichtweise isolierter ReLU-Test, MNIST
Methode (src/analog_geometry/isolated.py): Gleiches programmiertes und gedriftetes Netz wie im Sweep (gleiche noise_seeds).
Schicht l bekommt die DIGITALEN Aktivierungen von Schicht l-1 als Eingabe und wird mit der digitalen Schicht l bei gleicher Eingabe verglichen.
Gegenprobe „ideal“: max |Δ| <= 1.3e-5, CKA = 1.0 (MNIST und CIFAR). 5 Seeds × 3 Durchgänge je Drift-Zeit.
Ergebnis (results/isolated_summary_mnist.csv), cka_stimc:
- fc2 vor der ReLU: 0.9998 / 0.9998 / 0.9998 (1 d / 1 Wo / 1 J), Band-Min 0.9513 -> IM Band.
- fc2 nach der ReLU (act2): 0.9902 / 0.9857 / 0.9719, Band-Min 0.9617 -> IM Band (nicht außerhalb, wie die Lesart erwarten würde).
- fc3 (keine ReLU): 0.9997 / 0.9997 / 0.9996 -> IM Band.
- Schicht 1 ist ohnehin isoliert (Eingabe = Bild): vor der ReLU >= 0.9997 im Band, danach 1 Wo / 1 J außerhalb, 1 d unentschieden.
Bewertung: Die Lesart gilt isoliert NUR in Schicht 1. In Schicht 2 ist der eigene Effekt vor der ReLU ebenfalls ein reiner Versatz,
nach der ReLU sinkt CKA leicht, bleibt aber im Band. Die Abweichung von act2 im Gesamtnetz (z. B. 1 Wo 0.9477) ist damit
vereinbar mit einer Fortpflanzung aus Schicht 1, nicht mit einem eigenen ReLU-Effekt dieser Schicht.
Die vorgegebene Formulierung „Der Versatz ist pro Schicht additiv und wird erst durch die Nichtlinearität ... fortpflanzt.“
gilt NICHT, weil die Lesart nicht in allen Schichten zutrifft.

### Report-Formulierung ReLU-Teil (Entwurf, angepasst)
„Isoliert betrachtet (digitale Eingabe, analoge Schicht) wirkt die CMO-Drift in jeder untersuchten Schicht vor der Nichtlinearität
als reiner Versatz pro Stimulus (linear CKA mit Zentrierung pro Stimulus ≥ 0,9996). Nur in der ersten Schicht verschiebt die ReLU
diesen Versatz in eine Abweichung außerhalb des Seed-Referenzbands (ab 1 Woche). In der zweiten Schicht bleibt der isolierte Effekt
im Band. Ihre Abweichung im Gesamtnetz ist mit einer Fortpflanzung aus der ersten Schicht vereinbar.“
(Unter den untersuchten Einstellungen; MNIST-MLP. CIFAR folgt.)

## CIFAR-Restlauf abgeschlossen (2026-11-15, 20:07–23:37)
Alle 25 Einstellungen vollständig (5 Seeds × 3 Durchgänge). Commits 0d322e7, a69e8f9, 21b5c33, af991a2, 5f45e9d.
RATE-Zeilen (s/Lauf): 49.4 (Rausch 1/2/8; durch parallele MNIST-Rechnung gebremst), 49.9 (Drift + Prä), 30.5 (Stufen), 41.7, 30.9.
Ergebnis nach Vorab-Regel (results/findings_cifar10.csv): KEIN Befund. Accuracy-stabil (<= 1 pp) sind nur ideal, Rauschskala 0 (0.50 pp)
und Stufen 64 (0.24 pp) / inf. Dort liegt die Geometrie überall im Band. Schon der Ankerpunkt (Skala 1, 1 s) verliert 1.07 pp.
Drift: 1 h 19.5 pp, 1 d 30.0 pp, 1 Wo 35.9 pp, 1 J 45.1 pp. Kontrolle 2a (ohne Mittelwert): 2.24 / 3.20 pp, Geometrie im Band.

## Punkt 3 / Punkt 1 auf CIFAR (Prä-Aktivierungen, isolierter Test)
Prä-Aktivierungen im Gesamtnetz (results/preact_summary_cifar10.csv), cka_stimc, Band-Min conv1 0.9755 / conv2 0.9144:
- conv1 (Eingabe = Bild, also isoliert): 0.9998 (1 s), 0.9908 (1 h), 0.9759 (1 d), 0.9623 (1 Wo), 0.9260 (1 J) -> ab 1 Wo AUSSERHALB schon vor der ReLU.
  1 d liegt 0.0004 über der Bandgrenze (im Band).
- conv2: ab 1 h außerhalb (0.784 ... 0.453).
- Kontrolle 2a: vor und nach der ReLU im Band (>= 0.986).
Isolierter Test (results/isolated_summary_cifar10.csv), cka_stimc:
- conv2 vor der ReLU: 0.9217 / 0.8983 / 0.8504 (1 d / 1 Wo / 1 J), Band-Min 0.9144 -> im Band (1 d), AUSSERHALB (1 Wo, 1 J).
- conv2 nach der ReLU: 0.8198 / 0.7671 / 0.6479, Band-Min 0.9235 -> außerhalb.
- fc (keine ReLU): 0.9996 / 0.9995 / 0.9994 -> im Band.
Bewertung: Auf CIFAR ist der Drift-Effekt in den Faltungsschichten schon VOR der ReLU kein reiner Versatz pro Stimulus.
Die MNIST-Lesart überträgt sich nicht. Ungeprüfte Vermutung: Bei Faltung ist der Versatz Δw·Σ(Eingaben im Rezeptivfeld)
ortsabhängig (Bildinhalt, Zero-Padding am Rand). Die Zentrierung pro Stimulus entfernt nur eine Konstante pro Bild.
Für die vollverbundene fc-Schicht gilt die Lesart (reiner Versatz) wie auf MNIST.
Ohnehin kein Befund nach Vorab-Regel, weil die Accuracy bei allen Drift-Zeiten >= 1 h um mehr als 1 pp fällt.

## GATE vor Schritt 4 (2026-11-16): Physikalische Plausibilität des gleichförmigen Drift-Versatzes
Code (aihwkit 1.2.0 installiert = master @ 350ff34; reram.py, conductance.py, noise/base.py, tiles/inference.py byte-gleich nach CRLF):
1. Konverter: ReRamCMONoiseModel verdrahtet `SingleDeviceConductanceConverter` fest (reram.py:222). Kein Argument `g_converter`,
   obwohl der Docstring eines nennt (reram.py:196; Issue-Entwurf #8). EIN Gerät pro Gewicht, kein G+/G-.
2. Abbildung: `scaled = (w - w_min) * scale_ratio`, `g = scaled + g_min` mit scale_ratio = (g_max - g_min)/(w_max - w_min) (conductance.py:380).
   Positive und negative Gewichte liegen auf demselben Gerät; die Null sitzt mitten im Bereich (bei w_min/w_max symmetrisch ~48,6 µS).
3. Drift: `g_mean = g_prog + (-0.089) * ln(t) * drift_scale` (reram.py:306), unabhängig vom Leitwert (Docstring reram.py:180, aus Messungen),
   dann Streuung, Lese-Rauschen, clamp auf [g_min, g_max] (reram.py:317). Die Drift wirkt auf das EINE Gerät.
4. Rückrechnung: `w = w_min + (g - g_min)/scale_ratio` (conductance.py:395). Es gibt kein Referenzgerät, das mitdriftet:
   Die Null-Referenz (w_min, g_min) ist implizit ideal und driftfrei. Damit wird Δg 1:1 zu Δw = Δg/scale_ratio für ALLE Gewichte.
   Die Drift-Kompensation (ein skalarer Faktor pro Kachel) kann einen additiven Versatz nicht entfernen.
Empirisch (scripts/gate_drift_offset.py -> results/gate_drift_offset.csv; 5 Seeds, Durchgang 0, Drift 1 d, alle Kacheln):
- single_device: Δw_drift (gedriftet - programmiert) ist für positive, negative und ~0-Gewichte gleich, z. B. MNIST fc1
  −0,00805 / −0,00805 / −0,00805 (sd 0,0073), erwartet −0,00804. CIFAR conv2: −0,0124 / −0,0125 / −0,0123, erwartet −0,0124.
  In allen Kacheln stimmt der Mittelwert auf < 2 % mit −0,089·ln(t)/scale_ratio überein (~2–2,5 % von max|w|).
  -> Der Versatz wirkt auf das GEWICHT, nicht auf eine Differenz zweier Geräte.
- single_pair (G+ - G-, eigener Override, gleiche g-Grenzen): positiv −0,0048, negativ +0,0049, ~0 ≈ 0 (MNIST fc1) -> kein Versatz,
  sondern Betragsschrumpfung zur Null (das ungenutzte Gerät sitzt bei g_min und wird beim Clamp festgehalten). Die globale
  Drift-Kompensation gleicht das weitgehend aus (alpha 1,07–1,17; Rest-Mittelwert nach alpha ≤ 0,0012).
Abgleich mit Issue-Entwurf #6: Der Docstring „logische Null -> g_min“ stammt sichtlich vom Paar-Konverter. Er ist nicht die Ursache,
beschreibt aber genau die Semantik (Differenz zweier Geräte), unter der sich der gleichförmige Anteil aufheben würde.
EINORDNUNG: (b) Der gleichförmige Gewichtsversatz ist ein Artefakt der Konverter-Wahl des Modells (ein Gerät pro Gewicht mit
impliziter, driftfreier Null-Referenz). Physikalisch gemessen ist nur die leitwertunabhängige Relaxation der Einzelgeräte.
Ob daraus in echter Hardware ein Gewichtsversatz wird, hängt von der (nicht modellierten) Referenz-/Differenzschaltung ab.

### Zusatzlauf Differenzpaar (Gate (b)), Drift 1 d / 1 Wo / 1 J, 5 Seeds × 3 Durchgänge, Vorab-Regel unverändert
scripts/run_diffpair.sh -> results/diffpair_<ds>.csv, findings_diffpair_<ds>.csv, sweep_summary_diffpair_<ds>.csv.
Gegenprobe: Paar-Konverter ohne Rauschen = digital (MNIST seed 0: 0.9838 = 0.9838); Default-Pfad unverändert (Drift 1 d, seed 0, rep 0
bitgleich zur Sweep-CSV, 0.9765).
| Datensatz | Drift | Δ Acc (pp) Paar | CKA/RSA außerhalb (Paar) | min CKA / RSA über Schichten | Befund | zum Vergleich single_device |
| MNIST | 1 d  | 0,00 | – | 0,9987 / 0,9978 | nein | 0,49 pp, CKA alle Schichten außerhalb -> Befund „nur CKA“ |
| MNIST | 1 Wo | 0,02 | – | 0,9983 / 0,9972 | nein | 0,85 pp, Befund „nur CKA“ |
| MNIST | 1 J  | 0,02 | – | 0,9974 / 0,9956 | nein | 2,54 pp, kein Befund |
| CIFAR | 1 d  | 1,12 | – | 0,9927 / 0,9857 | nein | 30,0 pp |
| CIFAR | 1 Wo | 1,41 | – | 0,9908 / 0,9817 | nein | 35,9 pp |
| CIFAR | 1 J  | 2,16 | – | 0,9862 / 0,9710 | nein | 45,1 pp |
Folge: Die drei MNIST-Befunde („nur CKA“, Drift 1 d / 1 Wo, Default-Peripherie 1 d) hängen an der Konverter-Wahl. Mit Differenzpaar
verschwinden sie, ebenso der Großteil des CIFAR-Accuracy-Verlusts. Der CIFAR-Verlust mit Paar (1,1–2,2 pp) liegt in der Größenordnung
des Ankerpunkts ohne Drift (1,07 pp, single_device); die Geometrie bleibt im Band.
STOPP vor Schritt 4 (Anweisung Nutzer bei Einordnung (b)).

## Entscheidung nach dem Gate (Nutzer, 2026-11-16)
- Hauptlauf bleibt (A) Einzelgerät, wie vorab festgelegt. Im Report steht offen, dass die Konverter-Frage erst nachträglich aufkam.
- Drei Varianten: (A) Einzelgerät + driftfreie Referenz, (B) Einzelgerät + mitdriftende Referenz (= 2a), (C) Differenzpaar.
- Neue Kernaussage: Der Drift-Schaden hängt an der Konverter-Annahme. Die Geometrie ist die zweite Ebene.
- Volles Raster für (C) (Rauschen, Stufen; MNIST, CIFAR) über Nacht: scripts/run_diffpair_grid.sh, Commit nach jedem Sweep, nur diffpair-Dateien.
### (B) = 2a? (scripts/check_b_equivalence.py -> results/check_b_equivalence.csv)
Gleiche Zufallszahlen in A und 2a (der Mittelwert-Term verbraucht keine). Abweichung von (w_A − Δg/sr) gegenüber w_2a nur durch den Clamp auf [g_min, g_max]:
Anteil abweichender Gewichte (> 1e-4·max|w|) 0,2 % / 0,7 % (MNIST / CIFAR, 1 d), 3,4 % / 5,8 % (1 J); mittlere Abweichung ≤ 0,006 % von max|w|.
Nicht in 2a: das Eigenrauschen eines realen Referenzgeräts. -> „nahezu äquivalent“, nicht exakt.
### Quellenprüfung (Punkt 5)
Docstring: „Falcone et al. (In Review)“ (reram.py:172-173), ohne Titel. Vermutete Veröffentlichung (eigene Zuordnung): Falcone et al.,
„All-in-One Analog AI Hardware: On-Chip Training and Inference with CMO/HfOx ReRAM Devices“, Adv. Funct. Mater. 2025, arXiv:2502.04524.
Die Arbeit sagt nur „pre-trained normalized weights are initially mapped into target conductances“ (2.2.1) bzw. „each weight in the normalized
matrix (ranging from [-1, 1]) is mapped to its corresponding conductance value“ (4.6.1). Einzelgerät oder Paar, Referenz: nicht angegeben -> OFFEN.
Relaxation gemessen bis 1 Woche (2.2.2), 10 Jahre dort extrapoliert -> unser 1-J-Punkt ist Extrapolation.
