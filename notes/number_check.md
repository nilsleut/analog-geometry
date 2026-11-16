# Zahlenprüfung report/report.md und report/summary_en.md -> results/*.csv

Erzeugt von `scripts/check_report_numbers.py` (feste Zuordnung Zahl -> CSV-Abfrage, in Reihenfolge des Texts).
Geprüft: jede Dezimalzahl, Vorzeichen ignoriert, gerundet auf die angezeigten Nachkommastellen (Report: Dezimalkomma; englische Zusammenfassung: Dezimalpunkt, Versionsnummern ausgenommen).
Nicht geprüft: Ganzzahlen (Seeds, Paare, Stufen, Skalen, Zeilennummern im Code).
Bei „Obergrenze ≤“ bzw. „Untergrenze ≥“ steht im Text eine Schranke; geprüft wird der gerundete Extremwert.

## report/report.md

**Ergebnis: 122 Zahlen, 0 Probleme.**

| Zeile | Zahl | Status | Datei | Abfrage = Wert | Kontext |
|---|---|---|---|---|---|
| 7 | 2,5 | ok | gate_drift_offset.csv | (A) max_Kachel |Δw|/max|w| in % (Mittel über Seeds) = 2.53171 | örmiger Gewichtsversatz von etwa 2–2,5 % von max/w |
| 7 | 30,0 | ok | findings_all.csv | variant=A, cifar10, drift_time, {'t_inference': 86400.0} -> acc_drop_pp_mean = 30.022 | drift_offset]. Er kostet auf CIFAR 30,0 pp Accuracy |
| 7 | 1,1 | ok | findings_all.csv | variant=C, cifar10, drift_time, {'t_inference': 86400.0} -> acc_drop_pp_mean = 1.11733 | ccuracy. Mit Differenzpaar sind es 1,1 pp, ohne Ve |
| 7 | 2,2 | ok | findings_all.csv | variant=B (2a), cifar10, drift_no_mean, {'t_inference': 86400.0} -> acc_drop_pp_mean = 2.24267 | ,1 pp, ohne Versatz (Kontrolle 2a) 2,2 pp [finding |
| 7 | 1,07 | ok | findings_all.csv | variant=A, cifar10, prog_noise, {'prog_scale': 1.0} -> acc_drop_pp_mean = 1.06667 | der Ankerpunkt ohne Drift-Versatz 1,07 pp kostet; |
| 13 | 98,42 | ok | baseline_summary.csv | mnist test_acc_mean ×100 = 98.416 | MLP auf MNIST (784-256-256-10) mit 98,42 ± 0,09 % un |
| 13 | 0,09 | ok | baseline_summary.csv | mnist test_acc_std ×100 = 0.0896103 | MNIST (784-256-256-10) mit 98,42 ± 0,09 % und ein C |
| 13 | 76,37 | ok | baseline_summary.csv | cifar10 test_acc_mean ×100 = 76.374 | CIFAR-10 (2 Faltungen + 1 FC) mit 76,37 ± 0,45 % [b |
| 13 | 0,45 | ok | baseline_summary.csv | cifar10 test_acc_std ×100 = 0.451143 | 0 (2 Faltungen + 1 FC) mit 76,37 ± 0,45 % [baseline |
| 24 | 0,089 | ok | aihwkit reram.py:306 | decay_dict['mean'] = -0.08900206 = 0.0890021 | , leitwertunabhängige Relaxation `−0,089·ln t` µS (r |
| 30 | 0,2 | ok | check_b_equivalence.csv | mnist 1 d frac_differs ×100 (Mittel) = 0.224913 | unterscheiden sich nach 1 Tag bei 0,2 % (MNIST) b |
| 30 | 0,7 | ok | check_b_equivalence.csv | cifar10 1 d frac_differs ×100 (Mittel) = 0.69022 | nach 1 Tag bei 0,2 % (MNIST) bzw. 0,7 % (CIFAR) d |
| 30 | 3,4 | ok | check_b_equivalence.csv | mnist 1 J frac_differs ×100 (Mittel) = 3.40403 | FAR) der Gewichte, nach 1 Jahr bei 3,4 % bzw. 5,8 |
| 30 | 5,8 | ok | check_b_equivalence.csv | cifar10 1 J frac_differs ×100 (Mittel) = 5.83397 | wichte, nach 1 Jahr bei 3,4 % bzw. 5,8 %. Im Mitte |
| 30 | 0,006 | ok | check_b_equivalence.csv | max über (ds, t) mean_absdiff_rel ×100 (Obergrenze ≤) = 0.00554847 | m Mittel beträgt der Unterschied ≤ 0,006 % von max/w |
| 37 | 1,97 | ok | gate_drift_offset.csv | (A) min_Kachel |Δw|/max|w| in % = 1.96715 | d ≈ 0-Gewichte gleich. Er beträgt −1,97 bis −2,53 % |
| 37 | 2,53 | ok | gate_drift_offset.csv | (A) max_Kachel |Δw|/max|w| in % = 2.53171 | chte gleich. Er beträgt −1,97 bis −2,53 % von max/w |
| 37 | 0,089 | ok | aihwkit reram.py:306 | decay_dict['mean'] = -0.08900206 = 0.0890021 | hel und entspricht der Vorhersage −0,089·ln(t)/scale |
| 37 | 1,33 | ok | gate_drift_offset.csv | (C) w>0 Mittel Δw/max|w| in % = -1.32719 | schrumpft der Betrag symmetrisch (−1,33 % bei w > 0 |
| 37 | 1,33 | ok | gate_drift_offset.csv | (C) w<0 Mittel Δw/max|w| in % = 1.32762 | g symmetrisch (−1,33 % bei w > 0, +1,33 % bei w < 0 |
| 37 | 1,04 | ok | gate_drift_offset.csv | (C) min alpha = 1.03814 | on gleicht das weitgehend aus (α = 1,04–1,19). Gege |
| 37 | 1,19 | ok | gate_drift_offset.csv | (C) max alpha = 1.19396 | eicht das weitgehend aus (α = 1,04–1,19). Gegen den |
| 37 | 0,94 | ok | gate_drift_offset.csv | (A) min alpha = 0.944573 | rsatz bei (A) wirkt sie nicht (α = 0,94–1,00). |
| 37 | 1,00 | ok | gate_drift_offset.csv | (A) max alpha = 1.00162 | bei (A) wirkt sie nicht (α = 0,94–1,00). |
| 40 | 1,07 | ok | findings_all.csv | variant=A, cifar10, drift_time, {'t_inference': 1.0} -> acc_drop_pp_mean = 1.06667 | ** Der Accuracy-Verlust steigt von 1,07 pp (1 s) au |
| 40 | 19,5 | ok | findings_all.csv | variant=A, cifar10, drift_time, {'t_inference': 3600.0} -> acc_drop_pp_mean = 19.5067 | rlust steigt von 1,07 pp (1 s) auf 19,5 / 30,0 / 35 |
| 40 | 30,0 | ok | findings_all.csv | variant=A, cifar10, drift_time, {'t_inference': 86400.0} -> acc_drop_pp_mean = 30.022 | teigt von 1,07 pp (1 s) auf 19,5 / 30,0 / 35,9 / 45 |
| 40 | 35,9 | ok | findings_all.csv | variant=A, cifar10, drift_time, {'t_inference': 604800.0} -> acc_drop_pp_mean = 35.8887 | on 1,07 pp (1 s) auf 19,5 / 30,0 / 35,9 / 45,1 pp ( |
| 40 | 45,1 | ok | findings_all.csv | variant=A, cifar10, drift_time, {'t_inference': 31536000.0} -> acc_drop_pp_mean = 45.1273 | pp (1 s) auf 19,5 / 30,0 / 35,9 / 45,1 pp (1 h / 1 |
| 41 | 2,2 | ok | findings_all.csv | variant=B (2a), cifar10, drift_no_mean, {'t_inference': 86400.0} -> acc_drop_pp_mean = 2.24267 | CIFAR, (B) und (C):** (B) verliert 2,2 / 3,2 pp (1 |
| 41 | 3,2 | ok | findings_all.csv | variant=B (2a), cifar10, drift_no_mean, {'t_inference': 31536000.0} -> acc_drop_pp_mean = 3.20467 | (B) und (C):** (B) verliert 2,2 / 3,2 pp (1 d / 1 |
| 41 | 1,1 | ok | findings_all.csv | variant=C, cifar10, drift_time, {'t_inference': 86400.0} -> acc_drop_pp_mean = 1.11733 | iert 2,2 / 3,2 pp (1 d / 1 J), (C) 1,1 / 1,4 / 2,2 |
| 41 | 1,4 | ok | findings_all.csv | variant=C, cifar10, drift_time, {'t_inference': 604800.0} -> acc_drop_pp_mean = 1.40933 | ,2 / 3,2 pp (1 d / 1 J), (C) 1,1 / 1,4 / 2,2 pp (1 |
| 41 | 2,2 | ok | findings_all.csv | variant=C, cifar10, drift_time, {'t_inference': 31536000.0} -> acc_drop_pp_mean = 2.16 | ,2 pp (1 d / 1 J), (C) 1,1 / 1,4 / 2,2 pp (1 d / 1 |
| 42 | 1,07 | ok | findings_all.csv | variant=A, cifar10, prog_noise, {'prog_scale': 1.0} -> acc_drop_pp_mean = 1.06667 | -Versatz, weil ln 1 = 0) liegt mit 1,07 pp über der |
| 42 | 0,50 | ok | findings_all.csv | variant=A, cifar10, prog_noise, {'prog_scale': 0.0} -> acc_drop_pp_mean = 0.498667 | rogrammierrauschen (Rauschskala 0: 0,50 pp) oder mi |
| 43 | 0,28 | ok | findings_all.csv | variant=C, cifar10, prog_noise, {'prog_scale': 1.0} -> acc_drop_pp_mean = 0.277333 | ** Der Ankerpunkt von (C) verliert 0,28 pp. Die Dri |
| 43 | 1,1 | ok | findings_all.csv | variant=C, cifar10, drift_time, {'t_inference': 86400.0} -> acc_drop_pp_mean = 1.11733 | 28 pp. Die Drift-Punkte liegen mit 1,1 / 1,4 / 2,2 |
| 43 | 1,4 | ok | findings_all.csv | variant=C, cifar10, drift_time, {'t_inference': 604800.0} -> acc_drop_pp_mean = 1.40933 | Die Drift-Punkte liegen mit 1,1 / 1,4 / 2,2 pp (1 |
| 43 | 2,2 | ok | findings_all.csv | variant=C, cifar10, drift_time, {'t_inference': 31536000.0} -> acc_drop_pp_mean = 2.16 | rift-Punkte liegen mit 1,1 / 1,4 / 2,2 pp (1 d / 1 |
| 48 | 0,49 | ok | findings_all.csv | variant=A, mnist, drift_time, {'t_inference': 86400.0} -> acc_drop_pp_mean = 0.491333 | / Drift 1 d / 0,49 pp / act1 0 |
| 48 | 0,937 | ok | sweep_summary_mnist.csv | drift_time, act1, {'t_inference': 86400.0} -> cka_mean = 0.937299 | / Drift 1 d / 0,49 pp / act1 0,937 (0,956), ac |
| 48 | 0,956 | ok | reference_band_mnist.csv | act1 -> cka_min = 0.955687 | Drift 1 d / 0,49 pp / act1 0,937 (0,956), act2 0,95 |
| 48 | 0,954 | ok | sweep_summary_mnist.csv | drift_time, act2, {'t_inference': 86400.0} -> cka_mean = 0.954418 | 0,49 pp / act1 0,937 (0,956), act2 0,954 (0,962) auß |
| 48 | 0,962 | ok | reference_band_mnist.csv | act2 -> cka_min = 0.961739 | / act1 0,937 (0,956), act2 0,954 (0,962) außerhalb; |
| 48 | 0,955 | ok | sweep_summary_mnist.csv | drift_time, fc3, {'t_inference': 86400.0} -> cka_mean = 0.95526 | act2 0,954 (0,962) außerhalb; fc3 0,955 (0,956) une |
| 48 | 0,956 | ok | reference_band_mnist.csv | fc3 -> cka_min = 0.956156 | ,954 (0,962) außerhalb; fc3 0,955 (0,956) unentsch. |
| 49 | 0,85 | ok | findings_all.csv | variant=A, mnist, drift_time, {'t_inference': 604800.0} -> acc_drop_pp_mean = 0.846 | / Drift 1 Wo / 0,85 pp / act1 0 |
| 49 | 0,909 | ok | sweep_summary_mnist.csv | drift_time, act1, {'t_inference': 604800.0} -> cka_mean = 0.908884 | / Drift 1 Wo / 0,85 pp / act1 0,909, act2 0,945 |
| 49 | 0,945 | ok | sweep_summary_mnist.csv | drift_time, act2, {'t_inference': 604800.0} -> cka_mean = 0.944811 | 1 Wo / 0,85 pp / act1 0,909, act2 0,945, fc3 0,945 |
| 49 | 0,945 | ok | sweep_summary_mnist.csv | drift_time, fc3, {'t_inference': 604800.0} -> cka_mean = 0.944953 | 5 pp / act1 0,909, act2 0,945, fc3 0,945 außerhalb; |
| 50 | 0,67 | ok | findings_all.csv | variant=A, mnist, periphery_default, {} -> acc_drop_pp_mean = 0.665333 | / Default-Peripherie, 1 d / 0,67 pp / act1 0 |
| 50 | 0,935 | ok | sweep_summary_mnist.csv | periphery_default, act1, {} -> cka_mean = 0.934856 | t-Peripherie, 1 d / 0,67 pp / act1 0,935, act2 0,950 |
| 50 | 0,950 | ok | sweep_summary_mnist.csv | periphery_default, act2, {} -> cka_mean = 0.9503 | , 1 d / 0,67 pp / act1 0,935, act2 0,950, fc3 0,948 |
| 50 | 0,948 | ok | sweep_summary_mnist.csv | periphery_default, fc3, {} -> cka_mean = 0.948397 | 7 pp / act1 0,935, act2 0,950, fc3 0,948 außerhalb / |
| 52 | 0,0009 | ok | sweep_summary_mnist.csv | drift 1 d fc3 cka_band_min − cka_mean = 0.000895694 | 1 d liegt der Abstand zur Grenze (0,0009) innerhalb |
| 52 | 0,0055 | ok | sweep_summary_mnist.csv | drift_time, fc3, {'t_inference': 86400.0} -> cka_edge_tol = 0.00546797 | 9) innerhalb der Bootstrap-Breite (0,0055) [band_boot |
| 52 | 0,93 | ok | findings_all.csv | variant=Wan2022 (Anhang), mnist, wan2022_time, {'t_inference': 86400.0} -> acc_drop_pp_mean = 0.934 | g-Punkte Wan2022 nach 1 d und 2 d (0,93 / 0,92 pp; |
| 52 | 0,92 | ok | findings_all.csv | variant=Wan2022 (Anhang), mnist, wan2022_time, {'t_inference': 172800.0} -> acc_drop_pp_mean = 0.92 | e Wan2022 nach 1 d und 2 d (0,93 / 0,92 pp; nur RSA |
| 52 | 0,08 | ok | findings_all.csv | MNIST B + C(Drift): max acc_drop_pp_mean (Obergrenze ≤) = 0.0813333 | i Befunde: Der Verlust liegt bei ≤ 0,08 pp, CKA bei |
| 52 | 0,997 | ok | findings_all.csv | MNIST B + C(Drift): min cka_min_over_layers (Untergrenze ≥) = 0.997403 | ust liegt bei ≤ 0,08 pp, CKA bei ≥ 0,997 [findings_a |
| 55 | 0,990 | ok | sweep_summary_mnist.csv | drift_time, act1, {'t_inference': 86400.0} -> rsa_mean = 0.990035 | uf MNIST nicht unter dem Band. Mit 0,990 (act1, 1 d) |
| 55 | 0,925 | ok | reference_band_mnist.csv | act1 -> rsa_min = 0.925107 | höher als das Band der Seed-Paare (0,925–0,944). CKA |
| 55 | 0,944 | ok | reference_band_mnist.csv | act1 -> rsa_max = 0.944407 | als das Band der Seed-Paare (0,925–0,944). CKA fällt |
| 59 | 0,9996 | ok | isolated_summary_mnist.csv + preact_summary_mnist.csv | min cka_stimc vor ReLU (Untergrenze ≥) = 0.999589 | atz pro Stimulus (zentrierte CKA ≥ 0,9996). Nur in Sc |
| 60 | 0,898 | ok | isolated_summary_cifar10.csv | conv2 pre 1 Wo cka_stimc_mean = 0.89829 | ein reiner Versatz: conv2 isoliert 0,898 gegen Band- |
| 60 | 0,914 | ok | isolated_summary_cifar10.csv | conv2 pre cka_stimc_band_min = 0.914356 | onv2 isoliert 0,898 gegen Band-Min 0,914 nach 1 Wo [ |
| 63 | 11,1 | ok | findings_all.csv | variant=A, cifar10, prog_noise, {'prog_scale': 4.0} -> acc_drop_pp_mean = 11.12 | llt der fc-Layer aus dem Band, bei 11,1 pp Verlust. |
| 64 | 0,24 | ok | findings_all.csv | variant=A, mnist, prog_noise, {'prog_scale': 4.0} -> acc_drop_pp_mean = 0.235333 | ust bleibt bis Skala 4 unter 1 pp (0,24 pp). Bei Sk |
| 64 | 1,13 | ok | findings_all.csv | variant=A, mnist, prog_noise, {'prog_scale': 8.0} -> acc_drop_pp_mean = 1.13133 | pp (0,24 pp). Bei Skala 8 sind es 1,13 pp, bei 16 |
| 64 | 14,7 | ok | findings_all.csv | variant=A, mnist, prog_noise, {'prog_scale': 16.0} -> acc_drop_pp_mean = 14.6713 | ala 8 sind es 1,13 pp, bei 16 dann 14,7 pp mit alle |
| 65 | 18,3 | ok | findings_all.csv | variant=A, cifar10, levels, {'levels': '8'} -> acc_drop_pp_mean = 18.316 | - **Stufen, CIFAR:** 18,3 / 4,4 / 1,1 |
| 65 | 4,4 | ok | findings_all.csv | variant=A, cifar10, levels, {'levels': '16'} -> acc_drop_pp_mean = 4.404 | - **Stufen, CIFAR:** 18,3 / 4,4 / 1,15 / 0, |
| 65 | 1,15 | ok | findings_all.csv | variant=A, cifar10, levels, {'levels': '32'} -> acc_drop_pp_mean = 1.15 | - **Stufen, CIFAR:** 18,3 / 4,4 / 1,15 / 0,24 pp b |
| 65 | 0,24 | ok | findings_all.csv | variant=A, cifar10, levels, {'levels': '64'} -> acc_drop_pp_mean = 0.24 | ufen, CIFAR:** 18,3 / 4,4 / 1,15 / 0,24 pp bei 8 / |
| 66 | 0,46 | ok | findings_all.csv | variant=A, mnist, levels, {'levels': '8'} -> acc_drop_pp_mean = 0.462 | MNIST:** Schon 8 Stufen kosten nur 0,46 pp, die Geo |
| 67 | 0,25 | ok | findings_all.csv | variant=C, mnist, prog_noise, {'prog_scale': 16.0} -> acc_drop_pp_mean = 0.247333 | uster als (A), siehe unten. MNIST: 0,25 / 1,23 / 17 |
| 67 | 1,23 | ok | findings_all.csv | variant=C, mnist, prog_noise, {'prog_scale': 32.0} -> acc_drop_pp_mean = 1.228 | ls (A), siehe unten. MNIST: 0,25 / 1,23 / 17,4 pp b |
| 67 | 17,4 | ok | findings_all.csv | variant=C, mnist, prog_noise, {'prog_scale': 64.0} -> acc_drop_pp_mean = 17.3887 | siehe unten. MNIST: 0,25 / 1,23 / 17,4 pp bei Skal |
| 67 | 14,7 | ok | findings_all.csv | variant=A, mnist, prog_noise, {'prog_scale': 16.0} -> acc_drop_pp_mean = 14.6713 | 17,4 pp bei Skala 16 / 32 / 64 (A: 14,7 / 69,6 / 86 |
| 67 | 69,6 | ok | findings_all.csv | variant=A, mnist, prog_noise, {'prog_scale': 32.0} -> acc_drop_pp_mean = 69.56 | bei Skala 16 / 32 / 64 (A: 14,7 / 69,6 / 86,2 pp); |
| 67 | 86,2 | ok | findings_all.csv | variant=A, mnist, prog_noise, {'prog_scale': 64.0} -> acc_drop_pp_mean = 86.2213 | ala 16 / 32 / 64 (A: 14,7 / 69,6 / 86,2 pp); die Ge |
| 67 | 1,41 | ok | findings_all.csv | variant=C, cifar10, prog_noise, {'prog_scale': 4.0} -> acc_drop_pp_mean = 1.41067 | leibt bis Skala 16 im Band. CIFAR: 1,41 / 4,89 / 16 |
| 67 | 4,89 | ok | findings_all.csv | variant=C, cifar10, prog_noise, {'prog_scale': 8.0} -> acc_drop_pp_mean = 4.888 | is Skala 16 im Band. CIFAR: 1,41 / 4,89 / 16,2 pp b |
| 67 | 16,2 | ok | findings_all.csv | variant=C, cifar10, prog_noise, {'prog_scale': 16.0} -> acc_drop_pp_mean = 16.242 | a 16 im Band. CIFAR: 1,41 / 4,89 / 16,2 pp bei Skal |
| 67 | 11,1 | ok | findings_all.csv | variant=A, cifar10, prog_noise, {'prog_scale': 4.0} -> acc_drop_pp_mean = 11.12 | / 16,2 pp bei Skala 4 / 8 / 16 (A: 11,1 / 32,0 / 54 |
| 67 | 32,0 | ok | findings_all.csv | variant=A, cifar10, prog_noise, {'prog_scale': 8.0} -> acc_drop_pp_mean = 32.0013 | pp bei Skala 4 / 8 / 16 (A: 11,1 / 32,0 / 54,7 pp); |
| 67 | 54,7 | ok | findings_all.csv | variant=A, cifar10, prog_noise, {'prog_scale': 16.0} -> acc_drop_pp_mean = 54.676 | Skala 4 / 8 / 16 (A: 11,1 / 32,0 / 54,7 pp); fc fäl |
| 68 | 8,44 | ok | findings_all.csv | variant=C, cifar10, levels, {'levels': '8'} -> acc_drop_pp_mean = 8.44 | - **Variante (C), Stufen:** CIFAR 8,44 / 1,50 / 0, |
| 68 | 1,50 | ok | findings_all.csv | variant=C, cifar10, levels, {'levels': '16'} -> acc_drop_pp_mean = 1.498 | riante (C), Stufen:** CIFAR 8,44 / 1,50 / 0,44 / 0, |
| 68 | 0,44 | ok | findings_all.csv | variant=C, cifar10, levels, {'levels': '32'} -> acc_drop_pp_mean = 0.438 | (C), Stufen:** CIFAR 8,44 / 1,50 / 0,44 / 0,07 pp b |
| 68 | 0,07 | ok | findings_all.csv | variant=C, cifar10, levels, {'levels': '64'} -> acc_drop_pp_mean = 0.068 | ufen:** CIFAR 8,44 / 1,50 / 0,44 / 0,07 pp bei 8 / |
| 68 | 18,3 | ok | findings_all.csv | variant=A, cifar10, levels, {'levels': '8'} -> acc_drop_pp_mean = 18.316 | pp bei 8 / 16 / 32 / 64 Stufen (A: 18,3 / 4,4 / 1,1 |
| 68 | 4,4 | ok | findings_all.csv | variant=A, cifar10, levels, {'levels': '16'} -> acc_drop_pp_mean = 4.404 | 8 / 16 / 32 / 64 Stufen (A: 18,3 / 4,4 / 1,15 / 0, |
| 68 | 1,15 | ok | findings_all.csv | variant=A, cifar10, levels, {'levels': '32'} -> acc_drop_pp_mean = 1.15 | / 32 / 64 Stufen (A: 18,3 / 4,4 / 1,15 / 0,24 pp), |
| 68 | 0,24 | ok | findings_all.csv | variant=A, cifar10, levels, {'levels': '64'} -> acc_drop_pp_mean = 0.24 | 64 Stufen (A: 18,3 / 4,4 / 1,15 / 0,24 pp), außerh |
| 68 | 0,26 | ok | findings_all.csv | variant=C, mnist, levels, {'levels': '8'} -> acc_drop_pp_mean = 0.26 | ds nur RSA fc bei 8 Stufen. MNIST: 0,26 pp bei 8 St |
| 68 | 0,46 | ok | findings_all.csv | variant=A, mnist, levels, {'levels': '8'} -> acc_drop_pp_mean = 0.462 | n. MNIST: 0,26 pp bei 8 Stufen (A: 0,46 pp) [findin |
| 70 | 2,4 | ok | fairness_tiles.csv | ratio (min over tiles, seeds); ratio = A_prog1/C_prog1 = 2.43211 | s Programmierrauschen in (A) daher 2,4- bis 4,9-ma |
| 70 | 4,9 | ok | fairness_tiles.csv | ratio (max over tiles, seeds); ratio = A_prog1/C_prog1 = 4.90095 | mierrauschen in (A) daher 2,4- bis 4,9-mal so groß |
| 70 | 1,73 | ok | fairness_tiles.csv | mnist: r weighted by n_weights = 1.73306 | en, mit r = (w_max−w_min)/max/w/ = 1,73 (MNIST) bzw |
| 70 | 1,75 | ok | fairness_tiles.csv | cifar10: r weighted by n_weights = 1.75365 | −w_min)/max/w/ = 1,73 (MNIST) bzw. 1,75 (CIFAR), al |
| 71 | 3,54 | ok | findings_fairness_cifar10.csv | prog_noise, {'prog_scale': 2.13} -> acc_drop_pp_mean = 3.53667 | chem Gewichtsfehler wie (C): CIFAR 3,54 pp (Skala 2 |
| 71 | 2,13 | ok | fairness_match.csv | cifar10, s_C=8.0 -> s_A_matched = 2.13 | hler wie (C): CIFAR 3,54 pp (Skala 2,13) gegen 4,89 |
| 71 | 4,89 | ok | findings_all.csv | variant=C, cifar10, prog_noise, {'prog_scale': 8.0} -> acc_drop_pp_mean = 4.888 | : CIFAR 3,54 pp (Skala 2,13) gegen 4,89 pp; MNIST 0 |
| 71 | 0,89 | ok | findings_fairness_mnist.csv | prog_noise, {'prog_scale': 7.34} -> acc_drop_pp_mean = 0.886 | (Skala 2,13) gegen 4,89 pp; MNIST 0,89 / 10,4 pp ( |
| 71 | 10,4 | ok | findings_fairness_mnist.csv | prog_noise, {'prog_scale': 14.68} -> acc_drop_pp_mean = 10.3713 | 2,13) gegen 4,89 pp; MNIST 0,89 / 10,4 pp (Skala 7 |
| 71 | 7,34 | ok | fairness_match.csv | mnist, s_C=32.0 -> s_A_matched = 7.34 | 89 pp; MNIST 0,89 / 10,4 pp (Skala 7,34 / 14,68) ge |
| 71 | 14,68 | ok | fairness_match.csv | mnist, s_C=64.0 -> s_A_matched = 14.68 | MNIST 0,89 / 10,4 pp (Skala 7,34 / 14,68) gegen 1,23 |
| 71 | 1,23 | ok | findings_all.csv | variant=C, mnist, prog_noise, {'prog_scale': 32.0} -> acc_drop_pp_mean = 1.228 | 10,4 pp (Skala 7,34 / 14,68) gegen 1,23 / 17,4 pp. |
| 71 | 17,4 | ok | findings_all.csv | variant=C, mnist, prog_noise, {'prog_scale': 64.0} -> acc_drop_pp_mean = 17.3887 | (Skala 7,34 / 14,68) gegen 1,23 / 17,4 pp. Je nach |
| 71 | 1,79 | ok | fairness_sigma.csv | cifar10, s_C=8.0: per-tile matched s_A, min over tiles = 1.78874 | reicht die angeglichene Skala von 1,79 bis 3,23 (C |
| 71 | 3,23 | ok | fairness_sigma.csv | cifar10, s_C=8.0: per-tile matched s_A, max over tiles = 3.23071 | ie angeglichene Skala von 1,79 bis 3,23 (CIFAR); (C |
| 72 | 7,90 | ok | findings_fairness_cifar10.csv | levels, {'levels': '13'} -> acc_drop_pp_mean = 7.898 | :** CIFAR, (A) mit 13 / 27 Stufen: 7,90 / 1,61 pp g |
| 72 | 1,61 | ok | findings_fairness_cifar10.csv | levels, {'levels': '27'} -> acc_drop_pp_mean = 1.612 | AR, (A) mit 13 / 27 Stufen: 7,90 / 1,61 pp gegen (C |
| 72 | 8,44 | ok | findings_all.csv | variant=C, cifar10, levels, {'levels': '8'} -> acc_drop_pp_mean = 8.44 | 61 pp gegen (C) mit 8 / 16 Stufen: 8,44 / 1,50 pp; |
| 72 | 1,50 | ok | findings_all.csv | variant=C, cifar10, levels, {'levels': '16'} -> acc_drop_pp_mean = 1.498 | egen (C) mit 8 / 16 Stufen: 8,44 / 1,50 pp; MNIST 0 |
| 72 | 0,14 | ok | findings_fairness_mnist.csv | levels, {'levels': '13'} -> acc_drop_pp_mean = 0.142 | / 16 Stufen: 8,44 / 1,50 pp; MNIST 0,14 pp gegen 0, |
| 72 | 0,26 | ok | findings_all.csv | variant=C, mnist, levels, {'levels': '8'} -> acc_drop_pp_mean = 0.26 | ,44 / 1,50 pp; MNIST 0,14 pp gegen 0,26 pp bei 8 St |
| 83 | 0,9557 | ok | band_bootstrap_mnist.csv | post, act1, cka -> band_min_observed = 0.955687 | NIST) ein beobachtetes Minimum von 0,9557, 95 %-Inter |
| 83 | 0,9557 | ok | band_bootstrap_mnist.csv | post, act1, cka -> boot_ci_low_2.5 = 0.955687 | inimum von 0,9557, 95 %-Intervall [0,9557; 0,9631] [b |
| 83 | 0,9631 | ok | band_bootstrap_mnist.csv | post, act1, cka -> boot_ci_high_97.5 = 0.963131 | on 0,9557, 95 %-Intervall [0,9557; 0,9631] [band_boot |

## report/summary_en.md

**Ergebnis: 19 Zahlen, 0 Probleme.**

| Zeile | Zahl | Status | Datei | Abfrage = Wert | Kontext |
|---|---|---|---|---|---|
| 8 | 98.42 | ok | baseline_summary.csv | mnist test_acc_mean ×100 = 98.416 | orks: MLP 784-256-256-10 on MNIST (98.42 % test accu |
| 8 | 76.37 | ok | baseline_summary.csv | cifar10 test_acc_mean ×100 = 76.374 | acy) and a 2-conv CNN on CIFAR-10 (76.37 %); 5 train |
| 16 | 2.5 | ok | gate_drift_offset.csv | (A) max_Kachel |Δw|/max|w| in % (Mittel über Seeds) = 2.53171 | a uniform weight offset of about 2–2.5 % of max/w/ |
| 16 | 0.94 | ok | gate_drift_offset.csv | (A) min alpha = 0.944573 | not remove an additive offset (α = 0.94–1.00). On C |
| 16 | 1.00 | ok | gate_drift_offset.csv | (A) max alpha = 1.00162 | emove an additive offset (α = 0.94–1.00). On CIFAR |
| 16 | 30.0 | ok | findings_all.csv | variant=A, cifar10, drift_time, {'t_inference': 86400.0} -> acc_drop_pp_mean = 30.022 | = 0.94–1.00). On CIFAR this costs 30.0 pp after 1 |
| 16 | 1.1 | ok | findings_all.csv | variant=C, cifar10, drift_time, {'t_inference': 86400.0} -> acc_drop_pp_mean = 1.11733 | costs 30.0 pp after 1 day, versus 1.1 pp with a d |
| 16 | 2.2 | ok | findings_all.csv | variant=B (2a), cifar10, drift_no_mean, {'t_inference': 86400.0} -> acc_drop_pp_mean = 2.24267 | p with a differential pair (C) and 2.2 pp with a d |
| 17 | 0.49 | ok | findings_all.csv | variant=A, mnist, drift_time, {'t_inference': 86400.0} -> acc_drop_pp_mean = 0.491333 | nder (A), MNIST drift after 1 day (0.49 pp) and 1 w |
| 17 | 0.85 | ok | findings_all.csv | variant=A, mnist, drift_time, {'t_inference': 604800.0} -> acc_drop_pp_mean = 0.846 | after 1 day (0.49 pp) and 1 week (0.85 pp) and the |
| 17 | 0.67 | ok | findings_all.csv | variant=A, mnist, periphery_default, {} -> acc_drop_pp_mean = 0.665333 | ) and the default-periphery point (0.67 pp) meet th |
| 17 | 0.937 | ok | sweep_summary_mnist.csv | drift_time, act1, {'t_inference': 86400.0} -> cka_mean = 0.937299 | r example, CKA in layer 1 drops to 0.937 against a b |
| 17 | 0.956 | ok | reference_band_mnist.csv | act1 -> cka_min = 0.955687 | to 0.937 against a band minimum of 0.956. Under (B) |
| 17 | 0.08 | ok | findings_all.csv | MNIST B + C(Drift): max acc_drop_pp_mean (Obergrenze ≤) = 0.0813333 | nd (C) all three disappear (loss ≤ 0.08 pp, CKA ≥ 0 |
| 17 | 0.997 | ok | findings_all.csv | MNIST B + C(Drift): min cka_min_over_layers (Untergrenze ≥) = 0.997403 | e disappear (loss ≤ 0.08 pp, CKA ≥ 0.997). On CIFAR |
| 17 | 1.07 | ok | findings_all.csv | variant=A, cifar10, prog_noise, {'prog_scale': 1.0} -> acc_drop_pp_mean = 1.06667 | without drift offset already loses 1.07 pp; for (C) |
| 18 | 0.990 | ok | sweep_summary_mnist.csv | drift_time, act1, {'t_inference': 86400.0} -> rsa_mean = 0.990035 | MNIST does not leave the band; at 0.990 (layer 1, 1 |
| 18 | 0.925 | ok | reference_band_mnist.csv | act1 -> rsa_min = 0.925107 | is even above the seed-pair band (0.925–0.944), whi |
| 18 | 0.944 | ok | reference_band_mnist.csv | act1 -> rsa_max = 0.944407 | en above the seed-pair band (0.925–0.944), while CKA |
