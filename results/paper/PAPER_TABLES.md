# Paper tables (regenerated from results/)


## Protocol of HybridXAI (CERT r4.2, chronological 80/20)

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | FPR |
|---|---|---|---|---|---|---|
| HybridXAI (reported) | 0.976 | NR | 0.89 | 0.86 | 0.87 | NR |
| HybridXAI (re-run) | 0.95240 ± 0.00507 | 0.123 ± 0.023 | 0.200 ± 0.044 | 0.549 ± 0.035 | 0.290 ± 0.043 | 0.00601 ± 0.00215 |
| RF (re-run) | 0.99999 ± 0.00000 | 0.998 ± 0.000 | 0.994 ± 0.006 | 0.933 ± 0.019 | 0.962 ± 0.009 | 0.00002 ± 0.00002 |
| ConFog | 0.99998 ± 0.00002 | 0.994 ± 0.004 | 0.952 ± 0.024 | 0.938 ± 0.019 | 0.945 ± 0.008 | 0.00012 ± 0.00007 |


## Protocols of RF (PR-AUC)

| Protocol | Release | RF (reported) | RF (re-run) | RF (re-run) + context | ConFog |
|---|---|---|---|---|---|
| Chronological 70/30 | r4.2 | 0.666 | 0.997 ± 0.000 | 0.997 ± 0.000 | 0.991 ± 0.001 |
| Chronological 70/30 | r5.2 | 0.815 | 0.957 ± 0.008 | 0.964 ± 0.007 | 0.956 ± 0.016 |
| User-disjoint 70/30 | r4.2 | 0.550 | 0.960 ± 0.026 | 0.966 ± 0.020 | 0.975 ± 0.008 |
| User-disjoint 70/30 | r5.2 | 0.625 | 0.876 ± 0.012 | 0.886 ± 0.026 | 0.900 ± 0.015 |


## Transfer between releases (PR-AUC)

| Train -> test | RF (reported) | RF (re-run) | ConFog |
|---|---|---|---|
| r4.2 -> r5.2 | 0.250 | 0.600 ± 0.088 | 0.589 ± 0.098 |
| r5.2 -> r4.2 | 0.390 | 0.915 ± 0.032 | 0.931 ± 0.028 |
| r6.2 -> r4.2 | 0.015 | 0.739 ± 0.104 | 0.471 ± 0.166 |
| r6.2 -> r5.2 | 0.135 | 0.532 ± 0.029 | 0.192 ± 0.065 |


## Generalisation gap (within-release user-disjoint mean minus cross-release mean, PR-AUC)

RF (re-run): 0.221; ConFog: 0.392; RF (reported): 0.390


## Previously unseen insider scenarios (leave-one-scenario-out)

| Release | Scenario | RF PR-AUC | ConFog PR-AUC | RF recall | ConFog recall |
|---|---|---|---|---|---|
| r4.2 | 1 | 0.464 ± 0.053 | 0.381 ± 0.040 | 0.489 ± 0.156 | 0.501 ± 0.149 |
| r4.2 | 2 | 0.562 ± 0.023 | 0.085 ± 0.151 | 0.000 ± 0.000 | 0.003 ± 0.007 |
| r4.2 | 3 | 0.483 ± 0.026 | 0.525 ± 0.025 | 0.420 ± 0.110 | 0.490 ± 0.022 |
| r5.2 | 1 | 0.397 ± 0.058 | 0.416 ± 0.054 | 0.386 ± 0.112 | 0.336 ± 0.129 |
| r5.2 | 2 | 0.411 ± 0.075 | 0.083 ± 0.115 | 0.003 ± 0.004 | 0.000 ± 0.000 |
| r5.2 | 3 | 0.148 ± 0.049 | 0.362 ± 0.167 | 0.230 ± 0.249 | 0.460 ± 0.042 |
| r5.2 | 4 | 0.457 ± 0.197 | 0.144 ± 0.118 | 0.000 ± 0.000 | 0.000 ± 0.000 |


## Ablation, CERT r4.2 chronological 80/20

| Variant | PR-AUC | F1 | Forwarded |
|---|---|---|---|
| ConFog (full) | 0.994 ± 0.004 | 0.945 ± 0.008 | 0.0026 ± 0.0001 |
| Fog tier only | 0.994 ± 0.004 | 0.960 ± 0.015 | 0.0000 ± 0.0000 |
| Without graph branch | 0.994 ± 0.004 | 0.946 ± 0.022 | 0.0026 ± 0.0001 |
| Fixed alert threshold | 0.993 ± 0.005 | 0.959 ± 0.014 | 0.0026 ± 0.0001 |
| Without context features | 0.989 ± 0.001 | 0.927 ± 0.017 | 0.0027 ± 0.0001 |
| Without days of history | 0.990 ± 0.002 | 0.940 ± 0.028 | 0.0025 ± 0.0000 |
| Without web keyword features | 0.991 ± 0.004 | 0.925 ± 0.023 | 0.0025 ± 0.0000 |
| Without IF and AE scores | 0.991 ± 0.004 | 0.925 ± 0.008 | 0.0025 ± 0.0001 |
| Cloud tier only (deep baseline) | 0.973 ± 0.005 | 0.907 ± 0.012 | 1.0000 ± 0.0000 |
| Cloud tier only, without graph | 0.973 ± 0.003 | 0.916 ± 0.009 | 1.0000 ± 0.0000 |
| Without logon logs | 0.990 ± 0.004 | 0.919 ± 0.018 | 0.0026 ± 0.0001 |
| Without file logs | 0.987 ± 0.007 | 0.935 ± 0.022 | 0.0026 ± 0.0002 |
| Without e-mail logs | 0.976 ± 0.005 | 0.934 ± 0.034 | 0.0026 ± 0.0001 |
| Without removable-device logs | 0.949 ± 0.025 | 0.860 ± 0.018 | 0.0051 ± 0.0040 |
| Without web logs | 0.516 ± 0.083 | 0.557 ± 0.025 | 0.6155 ± 0.3930 |
| Autoencoder only | 0.120 ± 0.017 | 0.267 ± 0.032 | 0.0000 ± 0.0000 |
| Isolation forest only | 0.007 ± 0.000 | 0.019 ± 0.001 | 0.0000 ± 0.0000 |


## Ablations and leakage checks under three protocols (PR-AUC)

| Variant | chrono80 r4.2 | userdisj70 r4.2 | userdisj70 r5.2 | loso r4.2 | loso r5.2 |
|---|---|---|---|---|---|
| ConFog (full) | 0.994 | 0.975 | 0.900 | 0.331 | 0.251 |
| Fog tier only | 0.994 | 0.980 | 0.896 | 0.416 | 0.283 |
| Without graph branch | 0.994 | 0.977 | 0.898 | 0.391 | 0.230 |
| Fixed alert threshold | 0.993 | 0.976 | 0.904 | 0.351 | 0.253 |
| Without context features | 0.989 | 0.970 | 0.895 | -- | -- |
| Without days of history | 0.990 | 0.974 | 0.890 | -- | -- |
| Without web keyword features | 0.991 | 0.974 | 0.887 | -- | -- |
| RF (re-run) | 0.998 | 0.960 | 0.876 | 0.503 | 0.353 |
| RF (re-run) with context features | 0.997 | 0.966 | 0.886 | -- | -- |
| Deep sequence baseline (GRU + graph, all records) | 0.973 | 0.906 | 0.766 | 0.108 | 0.041 |


## Overfitting check (train / validation / test PR-AUC)

| protocol | release | fog_train_pr_auc | fog_val_pr_auc | cloud_val_pr_auc | pr_auc |
|---|---|---|---|---|---|
| chrono70 | r4.2 | 1.000 ± 0.000 | 0.975 ± 0.020 | 0.935 ± 0.068 | 0.991 ± 0.001 |
| chrono70 | r5.2 | 1.000 ± 0.000 | 0.908 ± 0.039 | 0.748 ± 0.188 | 0.956 ± 0.016 |
| chrono80 | r4.2 | 1.000 ± 0.000 | 0.993 ± 0.003 | 0.996 ± 0.001 | 0.994 ± 0.004 |
| userdisj70 | r4.2 | 1.000 ± 0.000 | 0.962 ± 0.039 | 0.965 ± 0.036 | 0.975 ± 0.008 |
| userdisj70 | r5.2 | 1.000 ± 0.000 | 0.902 ± 0.056 | 0.876 ± 0.091 | 0.900 ± 0.015 |


## Paired bootstrap over test user-days

| Protocol | Release | Comparison | dPR-AUC | 95% CI | Seeds |
|---|---|---|---|---|---|
| chrono80 | r4.2 | ConFog - RF (re-run) | -0.0043 | [-0.0151, +0.0014] | 5 |
| chrono80 | r4.2 | ConFog - HybridXAI (re-run) | +0.8706 | [+0.8259, +0.9207] | 5 |
| chrono70 | r4.2 | ConFog - RF (re-run) | -0.0067 | [-0.0145, -0.0004] | 5 |
| chrono70 | r5.2 | ConFog - RF (re-run) | -0.0010 | [-0.0490, +0.0244] | 5 |
| userdisj70 | r4.2 | ConFog - RF (re-run) | +0.0154 | [-0.0125, +0.0547] | 5 |
| userdisj70 | r5.2 | ConFog - RF (re-run) | +0.0242 | [-0.0216, +0.0581] | 5 |


## Medical-device traffic (macro F1, MCC)

FogXGB reported on ToN-IoT 5-fold CV: {'accuracy': 1.0, 'f1_macro': 1.0, 'mcc': 1.0}

| dataset | protocol | model | f1_macro | mcc |
|---|---|---|---|---|
| ciciomt2024 | binary | cloud_lgbm | 0.978 ± 0.005 | 0.957 ± 0.009 |
| ciciomt2024 | binary | fog_lgbm | 0.976 ± 0.002 | 0.953 ± 0.005 |
| ciciomt2024 | binary | xgb_base15 | 0.969 ± 0.002 | 0.939 ± 0.004 |
| ciciomt2024 | category | cloud_lgbm | 0.782 ± 0.007 | 0.521 ± 0.001 |
| ciciomt2024 | category | fog_lgbm | 0.713 ± 0.031 | 0.505 ± 0.011 |
| ciciomt2024 | category | xgb_base15 | 0.778 ± 0.004 | 0.546 ± 0.001 |
| ciciomt2024 | device_insider | cloud_lgbm | 0.954 ± 0.002 | 0.909 ± 0.003 |
| ciciomt2024 | device_insider | fog_lgbm | 0.950 ± 0.003 | 0.901 ± 0.006 |
| ciciomt2024 | device_insider | xgb_base15 | 0.950 ± 0.002 | 0.901 ± 0.004 |
| ton_iot | cv5 | cloud_lgbm | 0.997 ± 0.001 | 0.993 ± 0.002 |
| ton_iot | cv5 | fog_lgbm | 0.997 ± 0.001 | 0.994 ± 0.001 |
| ton_iot | cv5 | xgb_base15 | 0.997 ± 0.001 | 0.993 ± 0.002 |
| ton_iot | holdout | cloud_lgbm | 0.998 ± 0.000 | 0.996 ± 0.001 |
| ton_iot | holdout | fog_lgbm | 0.998 ± 0.000 | 0.996 ± 0.000 |
| ton_iot | holdout | xgb_base15 | 0.998 ± 0.000 | 0.996 ± 0.001 |


## Testbed, 10,000-day sample

10,000 user-days, 3 fog nodes, docker (--cpus 1, --memory 512m, tc-netem), 3 repeats. Static mode: recall 22/25, Wilson 95% interval 0.70-0.96.

| scenario | mode | forward_rate | recall | decision_ms_p50 | decision_ms_p95 | alert_ms_p95 | uplink_kb | fog_cpu_pct | tau_mean |
|---|---|---|---|---|---|---|---|---|---|
| degraded | adaptive | 0.002 ± 0.000 | 0.840 ± 0.000 | 5.625 ± 0.081 | 29.151 ± 1.163 | 656.376 ± 2.889 | 8.905 ± 0.000 | 11.331 ± 0.216 | 0.986 ± 0.000 |
| normal | cloud_only | 1.000 ± 0.000 | -- | 20.474 ± 0.545 | 22.093 ± 1.116 | 22.093 ± 1.116 | 3414.355 ± 0.000 | 3.652 ± 0.226 | 0.000 ± 0.000 |
| overloaded_fog | static | 0.003 ± 0.000 | 0.880 ± 0.000 | 5.622 ± 0.122 | 27.512 ± 5.051 | 36.739 ± 1.880 | 9.260 ± 0.000 | 12.972 ± 0.118 | 0.502 ± 0.000 |
| congested_link | static | 0.003 ± 0.000 | 0.880 ± 0.000 | 7.232 ± 2.237 | 12.057 ± 1.331 | 661.834 ± 5.052 | 9.260 ± 0.000 | 14.933 ± 3.459 | 0.502 ± 0.000 |
| congested_link | cloud_only | 1.000 ± 0.000 | -- | 15340.959 ± 10.215 | 28857.470 ± 19.006 | 28857.470 ± 19.006 | 3414.355 ± 0.000 | 0.384 ± 0.016 | 0.000 ± 0.000 |
| degraded | cloud_only | 1.000 ± 0.000 | -- | 15246.917 ± 6.389 | 28674.342 ± 15.664 | 28674.342 ± 15.664 | 3414.355 ± 0.000 | 0.153 ± 0.031 | 0.000 ± 0.000 |
| normal | adaptive | 0.003 ± 0.000 | 0.880 ± 0.000 | 8.128 ± 2.817 | 9.520 ± 2.836 | 22.799 ± 5.285 | 9.260 ± 0.000 | 17.846 ± 5.557 | 0.502 ± 0.000 |
| congested_link | adaptive | 0.002 ± 0.000 | 0.840 ± 0.000 | 7.030 ± 1.347 | 11.284 ± 0.884 | 657.743 ± 1.554 | 8.905 ± 0.000 | 14.127 ± 2.256 | 0.986 ± 0.000 |
| overloaded_fog | adaptive | 0.003 ± 0.000 | 0.880 ± 0.000 | 5.870 ± 0.322 | 30.842 ± 8.097 | 35.604 ± 5.071 | 9.260 ± 0.000 | 13.709 ± 1.235 | 0.508 ± 0.004 |
| degraded | static | 0.003 ± 0.000 | 0.880 ± 0.000 | 5.572 ± 0.033 | 30.147 ± 1.388 | 680.580 ± 8.857 | 9.260 ± 0.000 | 11.262 ± 0.276 | 0.502 ± 0.000 |
| overloaded_fog | cloud_only | 1.000 ± 0.000 | -- | 17.027 ± 0.016 | 17.277 ± 0.101 | 17.277 ± 0.101 | 3414.355 ± 0.000 | 1.195 ± 0.294 | 0.000 ± 0.000 |
| normal | static | 0.003 ± 0.000 | 0.880 ± 0.000 | 6.594 ± 0.737 | 11.340 ± 1.787 | 25.674 ± 3.180 | 9.260 ± 0.000 | 15.962 ± 1.660 | 0.502 ± 0.000 |


## Testbed, full test set

63,275 user-days, 3 fog nodes, docker (--cpus 1, --memory 512m, tc-netem), 3 repeats. Static mode: recall 150/162, Wilson 95% interval 0.87-0.96.

| scenario | mode | forward_rate | recall | decision_ms_p50 | decision_ms_p95 | alert_ms_p95 | uplink_kb | fog_cpu_pct | tau_mean |
|---|---|---|---|---|---|---|---|---|---|
| degraded | adaptive | 0.002 ± 0.000 | 0.858 ± 0.000 | 5.472 ± 0.080 | 24.087 ± 2.962 | 705.081 ± 9.538 | 52.690 ± 0.000 | 18.754 ± 0.666 | 0.986 ± 0.000 |
| normal | cloud_only | 1.000 ± 0.000 | -- | 20.990 ± 0.517 | 22.964 ± 1.149 | 22.964 ± 1.149 | 21604.156 ± 0.000 | 5.320 ± 0.101 | 0.000 ± 0.000 |
| overloaded_fog | static | 0.003 ± 0.000 | 0.926 ± 0.000 | 7.581 ± 3.715 | 31.376 ± 13.867 | 47.552 ± 12.546 | 58.159 ± 0.000 | 22.742 ± 5.015 | 0.502 ± 0.000 |
| congested_link | static | 0.003 ± 0.000 | 0.926 ± 0.000 | 7.745 ± 3.156 | 9.248 ± 3.068 | 703.806 ± 7.640 | 58.159 ± 0.000 | 26.352 ± 9.632 | 0.502 ± 0.000 |
| congested_link | cloud_only | 1.000 ± 0.000 | -- | 95119.135 ± 44.071 | 180783.085 ± 54.981 | 180783.085 ± 54.981 | 21604.156 ± 0.000 | 0.378 ± 0.046 | 0.000 ± 0.000 |
| degraded | cloud_only | 1.000 ± 0.000 | -- | 94481.772 ± 17.345 | 179530.453 ± 36.744 | 179530.453 ± 36.744 | 21604.156 ± 0.000 | 0.135 ± 0.005 | 0.000 ± 0.000 |
| normal | adaptive | 0.003 ± 0.000 | 0.926 ± 0.000 | 8.065 ± 2.206 | 10.122 ± 2.314 | 23.855 ± 5.409 | 58.159 ± 0.000 | 28.461 ± 8.120 | 0.502 ± 0.000 |
| congested_link | adaptive | 0.002 ± 0.000 | 0.858 ± 0.000 | 6.896 ± 1.561 | 8.977 ± 3.229 | 698.876 ± 4.240 | 52.690 ± 0.000 | 23.749 ± 5.750 | 0.986 ± 0.000 |
| overloaded_fog | adaptive | 0.003 ± 0.000 | 0.926 ± 0.000 | 5.433 ± 0.034 | 23.968 ± 4.058 | 35.500 ± 8.117 | 58.041 ± 0.205 | 19.609 ± 0.449 | 0.502 ± 0.000 |
| degraded | static | 0.003 ± 0.000 | 0.926 ± 0.000 | 5.449 ± 0.041 | 21.346 ± 6.374 | 700.455 ± 3.318 | 58.159 ± 0.000 | 18.625 ± 0.046 | 0.502 ± 0.000 |
| overloaded_fog | cloud_only | 1.000 ± 0.000 | -- | 17.026 ± 0.034 | 17.214 ± 0.072 | 17.214 ± 0.072 | 21604.156 ± 0.000 | 1.711 ± 0.064 | 0.000 ± 0.000 |
| normal | static | 0.003 ± 0.000 | 0.926 ± 0.000 | 6.625 ± 1.537 | 9.308 ± 2.847 | 22.011 ± 3.068 | 58.159 ± 0.000 | 24.518 ± 4.969 | 0.502 ± 0.000 |


## Saturated gateway, default constants

30,000 user-days, 3 fog nodes, docker (--cpus 1, --memory 512m, tc-netem), 3 repeats. Static mode: recall 72/78, Wilson 95% interval 0.84-0.96.

| scenario | mode | forward_rate | recall | decision_ms_p50 | decision_ms_p95 | alert_ms_p95 | uplink_kb | fog_cpu_pct | tau_mean |
|---|---|---|---|---|---|---|---|---|---|
| saturated_degraded | adaptive | 0.002 ± 0.000 | 0.885 ± 0.000 | 1541.453 ± 141.685 | 3160.067 ± 682.239 | 3272.125 ± 277.619 | 26.445 ± 0.000 | 20.191 ± 0.736 | 0.986 ± 0.000 |
| saturated | adaptive | 0.002 ± 0.000 | 0.885 ± 0.000 | 1555.820 ± 161.042 | 3147.440 ± 439.352 | 2551.847 ± 50.118 | 26.445 ± 0.000 | 22.473 ± 1.360 | 0.983 ± 0.000 |
| saturated | static | 0.003 ± 0.000 | 0.923 ± 0.000 | 1483.407 ± 68.636 | 2969.544 ± 399.162 | 2952.168 ± 644.459 | 28.315 ± 0.000 | 22.371 ± 1.252 | 0.502 ± 0.000 |
| saturated_degraded | static | 0.003 ± 0.000 | 0.923 ± 0.000 | 1547.205 ± 223.242 | 3321.464 ± 1091.156 | 3419.724 ± 448.792 | 28.315 ± 0.000 | 19.959 ± 1.308 | 0.502 ± 0.000 |


## Saturated gateway, stricter constants

30,000 user-days, 3 fog nodes, docker (--cpus 1, --memory 512m, tc-netem), 3 repeats. Static mode: recall 72/78, Wilson 95% interval 0.84-0.96.

| scenario | mode | forward_rate | recall | decision_ms_p50 | decision_ms_p95 | alert_ms_p95 | uplink_kb | fog_cpu_pct | tau_mean |
|---|---|---|---|---|---|---|---|---|---|
| saturated_degraded | adaptive | 0.002 ± 0.000 | 0.885 ± 0.000 | 1606.222 ± 297.771 | 3204.942 ± 855.460 | 3746.621 ± 1083.151 | 26.445 ± 0.000 | 20.932 ± 0.283 | 0.986 ± 0.000 |
| saturated | adaptive | 0.002 ± 0.000 | 0.885 ± 0.000 | 1558.578 ± 84.588 | 2998.038 ± 300.116 | 2938.287 ± 389.577 | 26.445 ± 0.000 | 22.389 ± 0.912 | 0.983 ± 0.000 |
| saturated | static | 0.003 ± 0.000 | 0.923 ± 0.000 | 1678.654 ± 152.208 | 3461.610 ± 594.110 | 2793.956 ± 322.142 | 28.315 ± 0.000 | 21.452 ± 0.899 | 0.502 ± 0.000 |
| saturated_degraded | static | 0.003 ± 0.000 | 0.923 ± 0.000 | 1653.005 ± 332.771 | 3438.806 ± 1000.734 | 3628.966 ± 550.331 | 28.315 ± 0.000 | 20.569 ± 0.509 | 0.502 ± 0.000 |


## Saturated gateway, more lenient constants

30,000 user-days, 3 fog nodes, docker (--cpus 1, --memory 512m, tc-netem), 3 repeats. Static mode: recall 72/78, Wilson 95% interval 0.84-0.96.

| scenario | mode | forward_rate | recall | decision_ms_p50 | decision_ms_p95 | alert_ms_p95 | uplink_kb | fog_cpu_pct | tau_mean |
|---|---|---|---|---|---|---|---|---|---|
| saturated_degraded | adaptive | 0.002 ± 0.000 | 0.885 ± 0.000 | 1484.213 ± 111.247 | 2794.036 ± 211.344 | 3322.955 ± 350.197 | 26.445 ± 0.000 | 20.821 ± 0.745 | 0.984 ± 0.000 |
| saturated | adaptive | 0.002 ± 0.000 | 0.885 ± 0.000 | 1498.714 ± 33.658 | 2849.986 ± 143.947 | 2730.016 ± 226.271 | 26.445 ± 0.000 | 22.048 ± 0.869 | 0.982 ± 0.001 |
| saturated | static | 0.003 ± 0.000 | 0.923 ± 0.000 | 1545.448 ± 82.665 | 3069.719 ± 307.281 | 2882.835 ± 613.952 | 28.315 ± 0.000 | 21.806 ± 1.111 | 0.502 ± 0.000 |
| saturated_degraded | static | 0.003 ± 0.000 | 0.923 ± 0.000 | 1795.148 ± 507.533 | 3430.580 ± 717.909 | 3770.583 ± 591.731 | 28.315 ± 0.000 | 21.473 ± 0.397 | 0.502 ± 0.000 |


## Testbed energy, full test set (RAPL, idle power measured before each run)

63,275 user-days, 3 fog nodes, docker (--cpus 1, --memory 512m, tc-netem), 3 repeats. Static mode: recall 150/162, Wilson 95% interval 0.87-0.96.

| scenario | mode | forward_rate | recall | decision_ms_p50 | decision_ms_p95 | alert_ms_p95 | uplink_kb | fog_cpu_pct | tau_mean | total_energy_mwh_per_1k |
|---|---|---|---|---|---|---|---|---|---|---|
| degraded | adaptive | 0.002 ± 0.000 | 0.858 ± 0.000 | 5.819 ± 0.524 | 28.203 ± 4.285 | 706.776 ± 4.457 | 52.690 ± 0.000 | 19.607 ± 0.697 | 0.986 ± 0.000 | 0.951 ± 0.137 |
| normal | cloud_only | 1.000 ± 0.000 | -- | 20.975 ± 0.733 | 22.412 ± 1.100 | 22.412 ± 1.100 | 21604.156 ± 0.000 | 5.275 ± 0.102 | 0.000 ± 0.000 | 0.665 ± 0.337 |
| overloaded_fog | static | 0.003 ± 0.000 | 0.926 ± 0.000 | 5.465 ± 0.042 | 27.665 ± 2.025 | 33.345 ± 11.789 | 58.159 ± 0.000 | 19.521 ± 0.417 | 0.502 ± 0.000 | 1.010 ± 0.026 |
| congested_link | static | 0.003 ± 0.000 | 0.926 ± 0.000 | 6.856 ± 1.190 | 9.290 ± 2.831 | 702.651 ± 3.893 | 58.159 ± 0.000 | 24.161 ± 5.408 | 0.502 ± 0.000 | 1.572 ± 0.171 |
| congested_link | cloud_only | 1.000 ± 0.000 | -- | 95093.524 ± 73.265 | 180727.060 ± 82.160 | 180727.060 ± 82.160 | 21604.156 ± 0.000 | 0.390 ± 0.022 | 0.000 ± 0.000 | 10.468 ± 3.420 |
| degraded | cloud_only | 1.000 ± 0.000 | -- | 94492.567 ± 23.212 | 179556.126 ± 53.168 | 179556.126 ± 53.168 | 21604.156 ± 0.000 | 0.140 ± 0.012 | 0.000 ± 0.000 | 7.283 ± 0.255 |
| normal | adaptive | 0.003 ± 0.000 | 0.926 ± 0.000 | 9.174 ± 2.629 | 11.209 ± 2.546 | 27.093 ± 5.272 | 58.159 ± 0.000 | 32.030 ± 8.620 | 0.502 ± 0.000 | 1.345 ± 0.351 |
| congested_link | adaptive | 0.002 ± 0.000 | 0.858 ± 0.000 | 9.121 ± 3.196 | 11.315 ± 3.125 | 705.805 ± 8.030 | 52.690 ± 0.000 | 30.787 ± 9.094 | 0.986 ± 0.000 | 1.278 ± 0.248 |
| overloaded_fog | adaptive | 0.003 ± 0.000 | 0.926 ± 0.000 | 5.565 ± 0.126 | 24.571 ± 3.081 | 41.400 ± 11.296 | 58.159 ± 0.000 | 20.021 ± 0.585 | 0.503 ± 0.001 | 1.003 ± 0.095 |
| degraded | static | 0.003 ± 0.000 | 0.926 ± 0.000 | 5.483 ± 0.060 | 21.842 ± 7.450 | 710.340 ± 10.534 | 58.159 ± 0.000 | 18.703 ± 0.200 | 0.502 ± 0.000 | 0.996 ± 0.019 |
| overloaded_fog | cloud_only | 1.000 ± 0.000 | -- | 17.075 ± 0.052 | 17.353 ± 0.182 | 17.353 ± 0.182 | 21604.156 ± 0.000 | 1.734 ± 0.100 | 0.000 ± 0.000 | 0.587 ± 0.020 |
| normal | static | 0.003 ± 0.000 | 0.926 ± 0.000 | 5.971 ± 0.104 | 7.997 ± 1.543 | 20.505 ± 0.530 | 58.159 ± 0.000 | 21.781 ± 0.728 | 0.502 ± 0.000 | 1.806 ± 0.354 |


## Testbed at realistic load (20 records/s per node, 6000 user-days)

6,000 user-days, 3 fog nodes, docker (--cpus 1, --memory 512m, tc-netem), 3 repeats. Static mode: recall 12/14, Wilson 95% interval 0.60-0.96.

| scenario | mode | forward_rate | recall | decision_ms_p50 | decision_ms_p95 | alert_ms_p95 | uplink_kb | fog_cpu_pct | tau_mean |
|---|---|---|---|---|---|---|---|---|---|
| congested_link | adaptive | 0.002 ± 0.000 | 0.857 ± 0.000 | 9.935 ± 1.265 | 17.037 ± 4.367 | 337.383 ± 6.707 | 5.259 ± 0.000 | 0.428 ± 0.011 | 0.986 ± 0.000 |
| normal | cloud_only | 1.000 ± 0.000 | -- | 23.534 ± 0.395 | 27.200 ± 1.564 | 27.200 ± 1.564 | 2048.588 ± 0.000 | 0.104 ± 0.003 | 0.000 ± 0.000 |
| congested_link | cloud_only | 1.000 ± 0.000 | -- | 477.181 ± 0.377 | 480.472 ± 1.657 | 480.472 ± 1.657 | 2048.588 ± 0.000 | 0.102 ± 0.005 | 0.000 ± 0.000 |
| congested_link | static | 0.002 ± 0.000 | 0.857 ± 0.000 | 10.969 ± 0.452 | 16.173 ± 4.195 | 335.482 ± 2.880 | 5.259 ± 0.000 | 0.453 ± 0.033 | 0.502 ± 0.000 |
| normal | adaptive | 0.002 ± 0.000 | 0.857 ± 0.000 | 10.175 ± 0.987 | 12.793 ± 0.670 | 37.744 ± 3.037 | 5.259 ± 0.000 | 0.411 ± 0.032 | 0.502 ± 0.000 |
| normal | static | 0.002 ± 0.000 | 0.857 ± 0.000 | 10.646 ± 0.506 | 13.515 ± 1.994 | 36.350 ± 2.143 | 5.259 ± 0.000 | 0.439 ± 0.024 | 0.502 ± 0.000 |


## Ledger operations, single-node Hardhat

Node: Hardhat 2.22.17 (local, automine); tamper detected 100/100; security tests: {'unregistered_device_rejected': True, 'revoked_device_rejected': True, 'unauthorised_writer_rejected': True, 'replay_rejected': True, 'tampered_fog_model_rejected': True, 'firmware_mismatch_detected': True, 'firmware_match_accepted': True}

| Operation | Calls | Gas (mean) | Latency mean (ms) | Latency p95 (ms) |
|---|---|---|---|---|
| deploy_AccessPolicy | 1 | 213,357 | 15.89 | 15.89 |
| deploy_DeviceRegistry | 1 | 469,073 | 9.99 | 9.99 |
| deploy_AlertAnchor | 1 | 394,875 | 7.97 | 7.97 |
| set_writer | 3 | 45,616 | 8.69 | 9.19 |
| register_device | 50 | 74,246 | 7.57 | 9.61 |
| anchor_alert | 250 | 83,444 | 6.29 | 8.60 |
| anchor_batch | 5 | 83,691 | 5.89 | 6.28 |
| revoke_device | 1 | 33,771 | 7.91 | 7.91 |


## Ledger operations, four-validator Hyperledger Besu (QBFT)

Node: Hyperledger Besu (hyperledger/besu:24.9.1), QBFT, 4 validators, 1 s blocks, node and account permissioning; tamper detected 100/100; security tests: {'unregistered_device_rejected': True, 'revoked_device_rejected': True, 'unauthorised_writer_rejected': True, 'replay_rejected': True, 'tampered_fog_model_rejected': True, 'firmware_mismatch_detected': True, 'firmware_match_accepted': True, 'non_member_account_rejected': True, 'member_account_accepted': True}

| Operation | Calls | Gas (mean) | Latency mean (ms) | Latency p95 (ms) |
|---|---|---|---|---|
| deploy_AccessPolicy | 1 | 213,311 | 902.51 | 902.51 |
| deploy_DeviceRegistry | 1 | 468,941 | 953.44 | 953.44 |
| deploy_AlertAnchor | 1 | 394,759 | 1055.68 | 1055.68 |
| set_writer | 3 | 45,620 | 987.68 | 1042.60 |
| register_device | 50 | 74,246 | 999.48 | 1056.36 |
| anchor_alert | 250 | 83,444 | 999.85 | 1065.60 |
| anchor_batch | 5 | 83,691 | 1010.24 | 1068.34 |
| revoke_device | 1 | 33,771 | 523.52 | 523.52 |


## Fog model size, latency and accuracy

Rule fixed in advance: fastest configuration within 0.005 of the best validation PR-AUC. Mean over three seeds.

| Protocol | Configuration | Leaves | Anomaly scores | Size (KB) | Latency (ms/record) | Validation PR-AUC | Test PR-AUC |
|---|---|---|---|---|---|---|---|
| chrono80 | chosen on validation | 7 | False | 543 | 0.016 | 0.993 | 0.983 ± 0.012 |
| chrono80 | paper default | 31 | True | 2146 | 0.102 | 0.994 | 0.996 ± 0.002 |
| userdisj70 | chosen on validation | 7 | False | 389 | 0.011 | 0.985 | 0.970 ± 0.008 |
| userdisj70 | paper default | 31 | True | 1753 | 0.093 | 0.980 | 0.977 ± 0.006 |


## Insiders who imitate their normal behaviour (user-disjoint r4.2, five seeds)

| Model | alpha | Recall | PR-AUC |
|---|---|---|---|
| ConFog | 0.0 | 0.880 ± 0.099 | 0.976 ± 0.012 |
| ConFog | 0.25 | 0.830 ± 0.092 | 0.954 ± 0.017 |
| ConFog | 0.5 | 0.740 ± 0.097 | 0.906 ± 0.038 |
| ConFog | 0.75 | 0.540 ± 0.112 | 0.764 ± 0.058 |
| ConFog | 1.0 | 0.000 ± 0.000 | 0.006 ± 0.004 |
| RF (re-run) | 0.0 | 0.860 ± 0.073 | 0.960 ± 0.026 |
| RF (re-run) | 0.25 | 0.820 ± 0.072 | 0.945 ± 0.025 |
| RF (re-run) | 0.5 | 0.675 ± 0.060 | 0.881 ± 0.047 |
| RF (re-run) | 0.75 | 0.572 ± 0.046 | 0.740 ± 0.055 |
| RF (re-run) | 1.0 | 0.000 ± 0.000 | 0.010 ± 0.003 |


## Faithfulness of the alert reasons (161 forwarded days)

| Inputs replaced | Score drop, reasons | Score drop, random | Reasons larger | No longer forwarded (reasons / random) | Wilcoxon p |
|---|---|---|---|---|---|
| 1 | 0.973 | 0.025 | 100.0% | 99.4% / 0.0% | 1.8e-28 |
| 3 | 0.989 | 0.063 | 100.0% | 100.0% / 0.6% | 1.8e-28 |
| 5 | 0.989 | 0.105 | 100.0% | 100.0% / 1.2% | 1.8e-28 |
