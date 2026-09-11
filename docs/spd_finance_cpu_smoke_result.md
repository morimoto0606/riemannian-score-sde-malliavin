# SPD finance CPU smoke検証結果

## 結果

サーバーの実行ログでは、Hydra設定テスト2件が成功し、CPUの小規模end-to-end
smokeがVaradhan、ISM、Malliavin Hutchinsonの全3手法で完了している。
対象checkoutは `a654d19`、出力は
`results/spd_finance_smoke_oela6g1j/summary.json` である。[1]

最終PASSは、スクリプトが全3手法の学習、別プロセスでのcheckpoint復元、生成、
少数サンプル評価をエラーなしで終えたことを示す。個別JSONはまだ取得されていないため、
最小固有値や対称性誤差の数値をこのレポートでは推定しない。[1][2]

## 検証条件

| 項目 | 条件 |
|---|---|
| backend | CPU指定、CUDA_VISIBLE_DEVICES空 |
| precision | JAX x64有効 |
| dataset | Gitで配布した金融NPZを使用する既定パス |
| training seed | 0 |
| updates | 各手法1回 |
| batch / eval batch | 2 / 2 |
| forward GRW steps | 1 |
| hidden layers | [16,16] |
| Malliavin weighting | time_weighting=true、lambda=5 |
| 生成 | 各8件、reverse steps=4 |
| 評価 | 各側最大8件、plotなし |
| 制限 | 子プロセスごと600秒 |

これらは `--tiny` とスクリプトの明示overrideから確認できる条件であり、
本番モデルや本番forward step数での実行結果ではない。[2]

## PASSが意味すること

| 検査 | 確認範囲 |
|---|---|
| Hydra | 設定テスト2件がOK。各手法のtree/list/config表示も正常終了 |
| モデル・dataset・optimizer | 各学習プロセスが1 updateまで到達 |
| forward / loss / update | 今回の小規模CPU入力・seedで実行経路が完了 |
| checkpoint restore | 各手法で別プロセスから復元、checkpoint_step=1を検査 |
| 生成SPD | 各8件、non_spd_count=0を検査 |
| checkpoint保全 | 生成前後でtree.pklとarrays.npyのSHA256が一致 |
| 評価 | AIRM等の少数サンプル評価プロセスが正常終了 |

生成コードは非有限値、許容範囲を超える非対称性、非正の固有値を拒否する。
したがってPASSは、単にファイルが生成されたという確認より強い。ただし、独立した
数理テスト全件の成功、全seedでの成功、モデル品質や離散化収束を証明するものではない。[2][3]

## 残る検証

GPUでは過去のログに `libcusolver.so.11` ロード失敗と古いptxasの警告がある。
今回のCPU成功は、それらを修復したという証拠にはならない。
GPU障害の診断は [サーバー診断レポート](spd_finance_server_diagnosis.md) を参照する。[4]

本番のbatch 8、forward N=16、hidden=[128,128,128]での1-update検証と、
GPUでの学習・生成は残っている。Malliavinの高階微分に伴う計算時間・メモリも
小規模CPUテストだけから判断できない。100K学習の再開条件として、整合したGPU環境で
最小行列分解テストと本番shapeのsmokeを先に確認する。

追加の数理テストは同じサーバー環境でCPUを指定して実行できる。
これは既存テストのframe、AIRM写像、divergence、SPD(1) Malliavin符号、
dataset split独立性などを確認する。今回提示された2件のconfigテストとは別である。[5]

```bash
JAX_PLATFORM_NAME=cpu CUDA_VISIBLE_DEVICES="" \
python -m unittest discover -s tests -p test_spd_training.py -v
```

先に既存結果の具体値を確認するには、追加の学習を行わず次を実行する。

```bash
cat results/spd_finance_smoke_oela6g1j/summary.json
```

## Sources

1. サーバー実行ログ（会話内の非公開記録）：`861f2b4..a654d19` のpull、
   configテスト2件OK、3手法のChecking表示、最終PASSと出力パス。
2. [scripts/smoke_spd_finance.py](../scripts/smoke_spd_finance.py)：
   tiny条件、subprocess成功判定、step/count/SPD検査、checkpointハッシュ比較。
3. [riemannian_score_sde/spd_generation.py](../riemannian_score_sde/spd_generation.py)：
   生成行列の検査条件とmetadata作成。
4. [サーバー診断レポート](spd_finance_server_diagnosis.md)：
   過去のGPU障害ログとNVIDIA公式互換性資料。
5. [tests/test_spd_training.py](../tests/test_spd_training.py)：独立した数理・学習・datasetテスト。
