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
