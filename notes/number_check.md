# Zahlenprüfung report/report.md -> results/*.csv

Erzeugt von `scripts/check_report_numbers.py` (feste Zuordnung Zahl -> CSV-Abfrage, in Reihenfolge des Reports).
Geprüft: jede Dezimalzahl, Vorzeichen ignoriert, gerundet auf die angezeigten Nachkommastellen.
Nicht geprüft: Ganzzahlen (Seeds, Paare, Stufen, Skalen, Zeilennummern im Code).
Bei „Obergrenze ≤“ bzw. „Untergrenze ≥“ steht im Report eine Schranke; geprüft wird der gerundete Extremwert.

**Ergebnis: 74 Zahlen im Report, 0 Probleme.**

| Zeile | Zahl | Status | Datei | Abfrage = Wert | Kontext |
|---|---|---|---|---|---|
| 7 | 2,5 | ok | gate_drift_offset.csv | (A) max_Kachel |Δw|/max|w| in % (Mittel über Seeds) = 2.53171 | örmiger Gewichtsversatz von etwa 2–2,5 % von max/w |
| 7 | 30,0 | ok | findings_all.csv | variant=A, cifar10, drift_time, {'t_inference': 86400.0} -> acc_drop_pp_mean = 30.022 | drift_offset]. Er kostet auf CIFAR 30,0 pp Accuracy |
| 7 | 1,1 | ok | findings_all.csv | variant=C, cifar10, drift_time, {'t_inference': 86400.0} -> acc_drop_pp_mean = 1.11733 | ccuracy. Mit Differenzpaar sind es 1,1 pp, ohne Ve |
| 7 | 2,2 | ok | findings_all.csv | variant=B (2a), cifar10, drift_no_mean, {'t_inference': 86400.0} -> acc_drop_pp_mean = 2.24267 | ,1 pp, ohne Versatz (Kontrolle 2a) 2,2 pp [finding |
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
| 34 | 1,97 | ok | gate_drift_offset.csv | (A) min_Kachel |Δw|/max|w| in % = 1.96715 | d ≈ 0-Gewichte gleich. Er beträgt −1,97 bis −2,53 % |
| 34 | 2,53 | ok | gate_drift_offset.csv | (A) max_Kachel |Δw|/max|w| in % = 2.53171 | chte gleich. Er beträgt −1,97 bis −2,53 % von max/w |
| 34 | 0,089 | ok | aihwkit reram.py:306 | decay_dict['mean'] = -0.08900206 = 0.0890021 | hel und entspricht der Vorhersage −0,089·ln(t)/scale |
| 34 | 1,33 | ok | gate_drift_offset.csv | (C) w>0 Mittel Δw/max|w| in % = -1.32719 | schrumpft der Betrag symmetrisch (−1,33 % bei w > 0 |
| 34 | 1,33 | ok | gate_drift_offset.csv | (C) w<0 Mittel Δw/max|w| in % = 1.32762 | g symmetrisch (−1,33 % bei w > 0, +1,33 % bei w < 0 |
| 34 | 1,04 | ok | gate_drift_offset.csv | (C) min alpha = 1.03814 | on gleicht das weitgehend aus (α = 1,04–1,19). Gege |
| 34 | 1,19 | ok | gate_drift_offset.csv | (C) max alpha = 1.19396 | eicht das weitgehend aus (α = 1,04–1,19). Gegen den |
| 34 | 0,94 | ok | gate_drift_offset.csv | (A) min alpha = 0.944573 | rsatz bei (A) wirkt sie nicht (α = 0,94–1,00). |
| 34 | 1,00 | ok | gate_drift_offset.csv | (A) max alpha = 1.00162 | bei (A) wirkt sie nicht (α = 0,94–1,00). |
| 37 | 1,07 | ok | findings_all.csv | variant=A, cifar10, drift_time, {'t_inference': 1.0} -> acc_drop_pp_mean = 1.06667 | ** Der Accuracy-Verlust steigt von 1,07 pp (1 s) au |
| 37 | 19,5 | ok | findings_all.csv | variant=A, cifar10, drift_time, {'t_inference': 3600.0} -> acc_drop_pp_mean = 19.5067 | rlust steigt von 1,07 pp (1 s) auf 19,5 / 30,0 / 35 |
| 37 | 30,0 | ok | findings_all.csv | variant=A, cifar10, drift_time, {'t_inference': 86400.0} -> acc_drop_pp_mean = 30.022 | teigt von 1,07 pp (1 s) auf 19,5 / 30,0 / 35,9 / 45 |
| 37 | 35,9 | ok | findings_all.csv | variant=A, cifar10, drift_time, {'t_inference': 604800.0} -> acc_drop_pp_mean = 35.8887 | on 1,07 pp (1 s) auf 19,5 / 30,0 / 35,9 / 45,1 pp ( |
| 37 | 45,1 | ok | findings_all.csv | variant=A, cifar10, drift_time, {'t_inference': 31536000.0} -> acc_drop_pp_mean = 45.1273 | pp (1 s) auf 19,5 / 30,0 / 35,9 / 45,1 pp (1 h / 1 |
| 38 | 2,2 | ok | findings_all.csv | variant=B (2a), cifar10, drift_no_mean, {'t_inference': 86400.0} -> acc_drop_pp_mean = 2.24267 | CIFAR, (B) und (C):** (B) verliert 2,2 / 3,2 pp (1 |
| 38 | 3,2 | ok | findings_all.csv | variant=B (2a), cifar10, drift_no_mean, {'t_inference': 31536000.0} -> acc_drop_pp_mean = 3.20467 | (B) und (C):** (B) verliert 2,2 / 3,2 pp (1 d / 1 |
| 38 | 1,1 | ok | findings_all.csv | variant=C, cifar10, drift_time, {'t_inference': 86400.0} -> acc_drop_pp_mean = 1.11733 | iert 2,2 / 3,2 pp (1 d / 1 J), (C) 1,1 / 1,4 / 2,2 |
| 38 | 1,4 | ok | findings_all.csv | variant=C, cifar10, drift_time, {'t_inference': 604800.0} -> acc_drop_pp_mean = 1.40933 | ,2 / 3,2 pp (1 d / 1 J), (C) 1,1 / 1,4 / 2,2 pp (1 |
| 38 | 2,2 | ok | findings_all.csv | variant=C, cifar10, drift_time, {'t_inference': 31536000.0} -> acc_drop_pp_mean = 2.16 | ,2 pp (1 d / 1 J), (C) 1,1 / 1,4 / 2,2 pp (1 d / 1 |
| 39 | 0,50 | ok | findings_all.csv | variant=A, cifar10, prog_noise, {'prog_scale': 0.0} -> acc_drop_pp_mean = 0.498667 | llungen ohne Drift (Rauschskala 0: 0,50 pp; ≥ 64 St |
| 39 | 1,07 | ok | findings_all.csv | variant=A, cifar10, prog_noise, {'prog_scale': 1.0} -> acc_drop_pp_mean = 1.06667 | rpunkt (A, Skala 1, 1 s) liegt mit 1,07 pp knapp üb |
| 44 | 0,49 | ok | findings_all.csv | variant=A, mnist, drift_time, {'t_inference': 86400.0} -> acc_drop_pp_mean = 0.491333 | / Drift 1 d / 0,49 pp / act1 0 |
| 44 | 0,937 | ok | sweep_summary_mnist.csv | drift_time, act1, {'t_inference': 86400.0} -> cka_mean = 0.937299 | / Drift 1 d / 0,49 pp / act1 0,937 (0,956), ac |
| 44 | 0,956 | ok | reference_band_mnist.csv | act1 -> cka_min = 0.955687 | Drift 1 d / 0,49 pp / act1 0,937 (0,956), act2 0,95 |
| 44 | 0,954 | ok | sweep_summary_mnist.csv | drift_time, act2, {'t_inference': 86400.0} -> cka_mean = 0.954418 | 0,49 pp / act1 0,937 (0,956), act2 0,954 (0,962) auß |
| 44 | 0,962 | ok | reference_band_mnist.csv | act2 -> cka_min = 0.961739 | / act1 0,937 (0,956), act2 0,954 (0,962) außerhalb; |
| 44 | 0,955 | ok | sweep_summary_mnist.csv | drift_time, fc3, {'t_inference': 86400.0} -> cka_mean = 0.95526 | act2 0,954 (0,962) außerhalb; fc3 0,955 (0,956) une |
| 44 | 0,956 | ok | reference_band_mnist.csv | fc3 -> cka_min = 0.956156 | ,954 (0,962) außerhalb; fc3 0,955 (0,956) unentsch. |
| 45 | 0,85 | ok | findings_all.csv | variant=A, mnist, drift_time, {'t_inference': 604800.0} -> acc_drop_pp_mean = 0.846 | / Drift 1 Wo / 0,85 pp / act1 0 |
| 45 | 0,909 | ok | sweep_summary_mnist.csv | drift_time, act1, {'t_inference': 604800.0} -> cka_mean = 0.908884 | / Drift 1 Wo / 0,85 pp / act1 0,909, act2 0,945 |
| 45 | 0,945 | ok | sweep_summary_mnist.csv | drift_time, act2, {'t_inference': 604800.0} -> cka_mean = 0.944811 | 1 Wo / 0,85 pp / act1 0,909, act2 0,945, fc3 0,945 |
| 45 | 0,945 | ok | sweep_summary_mnist.csv | drift_time, fc3, {'t_inference': 604800.0} -> cka_mean = 0.944953 | 5 pp / act1 0,909, act2 0,945, fc3 0,945 außerhalb; |
| 46 | 0,67 | ok | findings_all.csv | variant=A, mnist, periphery_default, {} -> acc_drop_pp_mean = 0.665333 | / Default-Peripherie, 1 d / 0,67 pp / act1 0 |
| 46 | 0,935 | ok | sweep_summary_mnist.csv | periphery_default, act1, {} -> cka_mean = 0.934856 | t-Peripherie, 1 d / 0,67 pp / act1 0,935, act2 0,950 |
| 46 | 0,950 | ok | sweep_summary_mnist.csv | periphery_default, act2, {} -> cka_mean = 0.9503 | , 1 d / 0,67 pp / act1 0,935, act2 0,950, fc3 0,948 |
| 46 | 0,948 | ok | sweep_summary_mnist.csv | periphery_default, fc3, {} -> cka_mean = 0.948397 | 7 pp / act1 0,935, act2 0,950, fc3 0,948 außerhalb / |
| 48 | 0,0009 | ok | sweep_summary_mnist.csv | drift 1 d fc3 cka_band_min − cka_mean = 0.000895694 | 1 d liegt der Abstand zur Grenze (0,0009) innerhalb |
| 48 | 0,0055 | ok | sweep_summary_mnist.csv | drift_time, fc3, {'t_inference': 86400.0} -> cka_edge_tol = 0.00546797 | 9) innerhalb der Bootstrap-Breite (0,0055) [band_boot |
| 48 | 0,93 | ok | findings_all.csv | variant=Wan2022 (Anhang), mnist, wan2022_time, {'t_inference': 86400.0} -> acc_drop_pp_mean = 0.934 | g-Punkte Wan2022 nach 1 d und 2 d (0,93 / 0,92 pp; |
| 48 | 0,92 | ok | findings_all.csv | variant=Wan2022 (Anhang), mnist, wan2022_time, {'t_inference': 172800.0} -> acc_drop_pp_mean = 0.92 | e Wan2022 nach 1 d und 2 d (0,93 / 0,92 pp; nur RSA |
| 48 | 0,08 | ok | findings_all.csv | MNIST B + C(Drift): max acc_drop_pp_mean (Obergrenze ≤) = 0.0813333 | i Befunde: Der Verlust liegt bei ≤ 0,08 pp, CKA bei |
| 48 | 0,997 | ok | findings_all.csv | MNIST B + C(Drift): min cka_min_over_layers (Untergrenze ≥) = 0.997403 | ust liegt bei ≤ 0,08 pp, CKA bei ≥ 0,997 [findings_a |
| 51 | 0,990 | ok | sweep_summary_mnist.csv | drift_time, act1, {'t_inference': 86400.0} -> rsa_mean = 0.990035 | uf MNIST nicht unter dem Band. Mit 0,990 (act1, 1 d) |
| 51 | 0,925 | ok | reference_band_mnist.csv | act1 -> rsa_min = 0.925107 | höher als das Band der Seed-Paare (0,925–0,944). CKA |
| 51 | 0,944 | ok | reference_band_mnist.csv | act1 -> rsa_max = 0.944407 | als das Band der Seed-Paare (0,925–0,944). CKA fällt |
| 55 | 0,9996 | ok | isolated_summary_mnist.csv + preact_summary_mnist.csv | min cka_stimc vor ReLU (Untergrenze ≥) = 0.999589 | atz pro Stimulus (zentrierte CKA ≥ 0,9996). Nur in Sc |
| 56 | 0,898 | ok | isolated_summary_cifar10.csv | conv2 pre 1 Wo cka_stimc_mean = 0.89829 | ein reiner Versatz: conv2 isoliert 0,898 gegen Band- |
| 56 | 0,914 | ok | isolated_summary_cifar10.csv | conv2 pre cka_stimc_band_min = 0.914356 | onv2 isoliert 0,898 gegen Band-Min 0,914 nach 1 Wo [ |
| 59 | 11,1 | ok | findings_all.csv | variant=A, cifar10, prog_noise, {'prog_scale': 4.0} -> acc_drop_pp_mean = 11.12 | llt der fc-Layer aus dem Band, bei 11,1 pp Verlust. |
| 60 | 0,24 | ok | findings_all.csv | variant=A, mnist, prog_noise, {'prog_scale': 4.0} -> acc_drop_pp_mean = 0.235333 | ust bleibt bis Skala 4 unter 1 pp (0,24 pp). Bei Sk |
| 60 | 1,13 | ok | findings_all.csv | variant=A, mnist, prog_noise, {'prog_scale': 8.0} -> acc_drop_pp_mean = 1.13133 | pp (0,24 pp). Bei Skala 8 sind es 1,13 pp, bei 16 |
| 60 | 14,7 | ok | findings_all.csv | variant=A, mnist, prog_noise, {'prog_scale': 16.0} -> acc_drop_pp_mean = 14.6713 | ala 8 sind es 1,13 pp, bei 16 dann 14,7 pp mit alle |
| 61 | 18,3 | ok | findings_all.csv | variant=A, cifar10, levels, {'levels': '8'} -> acc_drop_pp_mean = 18.316 | - **Stufen, CIFAR:** 18,3 / 4,4 / 1,1 |
| 61 | 4,4 | ok | findings_all.csv | variant=A, cifar10, levels, {'levels': '16'} -> acc_drop_pp_mean = 4.404 | - **Stufen, CIFAR:** 18,3 / 4,4 / 1,15 / 0, |
| 61 | 1,15 | ok | findings_all.csv | variant=A, cifar10, levels, {'levels': '32'} -> acc_drop_pp_mean = 1.15 | - **Stufen, CIFAR:** 18,3 / 4,4 / 1,15 / 0,24 pp b |
| 61 | 0,24 | ok | findings_all.csv | variant=A, cifar10, levels, {'levels': '64'} -> acc_drop_pp_mean = 0.24 | ufen, CIFAR:** 18,3 / 4,4 / 1,15 / 0,24 pp bei 8 / |
| 62 | 0,46 | ok | findings_all.csv | variant=A, mnist, levels, {'levels': '8'} -> acc_drop_pp_mean = 0.462 | MNIST:** Schon 8 Stufen kosten nur 0,46 pp, die Geo |
| 73 | 0,9557 | ok | band_bootstrap_mnist.csv | post, act1, cka -> band_min_observed = 0.955687 | NIST) ein beobachtetes Minimum von 0,9557, 95 %-Inter |
| 73 | 0,9557 | ok | band_bootstrap_mnist.csv | post, act1, cka -> boot_ci_low_2.5 = 0.955687 | inimum von 0,9557, 95 %-Intervall [0,9557; 0,9631] [b |
| 73 | 0,9631 | ok | band_bootstrap_mnist.csv | post, act1, cka -> boot_ci_high_97.5 = 0.963131 | on 0,9557, 95 %-Intervall [0,9557; 0,9631] [band_boot |
