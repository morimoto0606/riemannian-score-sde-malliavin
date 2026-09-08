# 金融実データの SPD(5) 実験準備

2026-09-08 に取得・前処理を実行。学習、本評価、生成サンプリングは未実行。
指定された `/home/export/home/ymorimoto/github/riemannian-score-sde-malliavin`
はこの環境にないため、ワークスペースの同名リポジトリに実装した。
データ準備に続いて、AIRM の Varadhan / ISM / Malliavin Hutchinson の学習・生成経路を追加した。
実際の学習・生成・JAX テストはユーザーのサーバー側で行う。開発環境では未実行。
以下の「現在の学習・生成実装」以降が今回追加した実装の仕様。

## データと取得結果

Source: Yahoo Finance chart API (`query2.finance.yahoo.com/v8/finance/chart`)、
interval=1d。要求期間は 2008-01-01 から 2026-09-08 未満。
yfinance の追加依存を避け、curl で各系列を一度取得する。
生 JSON と URL、取得時刻、SHA256 を保存し、以後は通信なしで再生成できる。

|資産|ticker|今回応答の有効価格期間|採用価格|応答内の無効行数|共通期間の欠損数*|
|---|---|---|---|---:|---:|
|日本株 TOPIX ETF|1306.T|2009-01-05–2026-09-07|Adj Close|2|0|
|日本の広範な債券 NOMURA-BPI ETF|2510.T|2017-12-06–2026-09-07|Adj Close|1|1|
|USD/JPY（円/米ドル）|JPY=X|2008-01-01–2026-09-07|Close|29|4|
|EUR/JPY（円/ユーロ）|EURJPY=X|2008-01-01–2026-09-07|Close|28|3|
|円建て現物金信託|1540.T|2010-07-02–2026-09-07|Adj Close|2|1|

*TOPIX の観測営業日に合わせ、FX を1配信営業日遅らせた後。下記の隔離2価格は別計上。
全5系列に Adj Close フィールドがある。FX は配当調整を必要としないので Close を使用。
各系列の有効観測数、平日基準の欠損数（祝日を含む）、タイムゾーン、価格停滞数は
NPZ 内の `metadata_json` と隣の `.metadata.json` に保存している。
2008年からの5資産共通期間は、この組み合わせでは確保できない。

債券の `2510.T` は純粋な JGB ETF ではない。国債に加え地方債・政府保証債・事業債等を含む
[運用会社の国内債券 ETF](https://nextfunds.jp/lineup/2510/)を、長めの履歴を優先した proxy とした。
[上場の公式発表](https://nextfunds.jp/data/2017/20171211_8DEBA3E7.pdf)に合わせ、
Yahoo にある上場前3観測を除外し **2017-12-11** から利用する。
純粋な日本国債の候補 `2561.T` も取得確認済み（今回の有効応答は
2020-02-21–2026-09-07、Adj Close あり、無効行1）。商品内容は
[BlackRock のファクトシート](https://www.blackrock.com/jp/individual/ja/literature/fact-sheet/2561-ishares-core-japan-government-bond-etf-fund-fact-sheet-ja-jp.pdf)
を参照。純国債に限定する場合は、こちらへ変更し共通期間を短くする必要がある。

Gold は [1540 の発行者資料](https://www.tr.mufg.jp/ippan/release/pdf_mutb/100614_1.pdf)
に基づく現物国内保管型信託を採用。Gold futures proxy と異なり、先物ロールや USD→JPY
変換の追加設計を避けられる。ETF の市場価格にはプレミアム・ディスカウントや流動性の
影響が残る。債券系列は共通期間で価格差ゼロが386回あり、NAV/債券指数そのものの
共分散とは異なる。

## 前処理・品質管理

1. Yahoo の日次ラベルを各取引所タイムゾーンで日付化し、指定終了日以後の live bar を除外する。
   FX の live bar は重複していたため、期間外を除外してから重複を検査する。
2. ETF は調整価格必須。欠落時に Close へ自動 fallback しない。非正・非有限値は欠損扱い。
3. 東京 ETF 終値に対して同日 FX 終値は後になるため、FX は1配信営業日前の値を使用。
   `price_observation_dates` に実際の FX 観測日を保存。全行で東京の対象日より前であることを確認。
   厳密な同時刻5資産価格ではなく、時差を明示した市場状態 proxy である。
4. TOPIX の有効な日次観測を基準カレンダーにし、共通履歴 2017-12-11–2026-09-07 の
   **2148営業日**へ reindex する。公式取引所カレンダーを使った完全性保証ではない。
5. 補間、forward fill、back fill、winsorize、標準化、年率換算はしない。
6. `r_t = log(P_t / P_{t-1})` を欠損行を除く**前**に計算する。欠損を飛ばして
   数営業日分を1日 return にしない。60連続基準営業日の return がすべて有限な窓だけ採用。
7. `Sigma_t = sum((r-rbar)(r-rbar)^T) / 59`。matrix size は5、内在次元は15。

品質上の具体的な処置は `config/dataset/spd_finance_quality.json` に記録した。
`1306.T` の 2026-03-30, 03-31 の Adj Close は 36.930489, 36.439915 で、
前後の 375.486145, 381.863617 と比べ余分な1/10調整が疑われる。
[発行者は4月1日効力の1:10分割を公表](https://nextfunds.jp/data/2026/td_260217d.pdf)している。
誤調整という判断は取得値と分割情報からの推定であり、正しい価格を推測して上書きはしない。
当該2日が異常レンジ [30,45] にある間だけ価格を NaN とし、関係する窓を除外する。
再取得で正常値に直った場合は隔離しない。将来の新しい絶対 log return >0.5 は build を
停止させ、データ確認を要求する。市場の通常の大変動まで自動削除しない。

候補2088窓から、欠損・隔離価格を含む287窓を除外し **1801窓**を保存。
データは取得時点の歴史的調整価格の snapshot であり、当時の配信内容を再現する
point-in-time backtest データではない。Yahoo の将来の改訂で再ダウンロード結果は変わり得る。
予測バックテストには適さず、今回は固定された共分散分布の生成実験用。

## SPD 検査と保存

|検査|結果|
|---|---:|
|shape / dtype|`(1801, 5, 5)` / float64|
|最大非対称誤差|0|
|全固有値の最小値|9.388683657654598e-7|
|全固有値の最大値|6.934594891828138e-4|
|determinant 範囲|8.860616025484818e-26 – 7.131116749526588e-21|
|最大 / 中央条件数|147.5478732938451 / 38.62023168798083|
|SPD|全1801行|
|regularization|なし、全行 eps=0|

検査は対称性、eigvalsh、det、slogdet、条件数を使用。財務データの小さな determinant は
行列のスケールにも依存するため、単独では正則化理由にしない。
デフォルトは `lambda_min < 1e-12 * lambda_max` で停止する。
明示的な `--allow-regularization` 指定時のみ、行ごとに
`eps=max(0,1e-12*lambda_max-lambda_min)` を対角に足し、前後の固有値と eps を保存する。
大きな負固有値・全ゼロ分散は修復しない。
geomstats の `projection` は `gs.atol` まで固有値を持ち上げるため、実データには呼んでいない。

保存先: `data/spd_finance/spd_finance_5asset_60d.npz`（約458 KB）。
主要キーは `covariances`, `dates`, `returns`, `return_dates`, `prices`, `price_dates`,
`price_observation_dates`, `tickers`, `asset_names`, `window_start_return_index`,
`window_end_return_index`, `train_indices`, `val_indices`, `test_indices`,
`eigenvalues`, `determinants`, `condition_numbers`, `regularization_eps`, `metadata_json`。
`dates` は共分散窓の最終日。`returns` は基準カレンダーの全2147行を保持し、
欠損を含み得る。保存された各 covariance の参照 return 範囲は有限である。
生 JSON、NPZ、JSON metadata は `/data/spd_finance/` ごと gitignore 対象。

## 時系列 split と seed

全共分散を古い順に70/15/15%へ分けてから、val/test の先頭を purge する。
return だけでなく境界価格も共有しないよう、通常60共分散窓ずつを除外する。
合計120行は NPZ に残るが、どの split にも所属させない。

|split|サンプル数|共分散日の範囲|
|---|---:|---|
|train|1260|2018-03-05–2023-07-18|
|val|210|2023-10-16–2024-08-22|
|test|211|2024-11-21–2026-09-07|

`SPDFinanceDataset` は保存済み index、日付順、SPD、境界非重複を検査する。
`run.py` は `chronological_splits` を持つ dataset ではそのメソッドを使い、
既存 dataset では従来の `random_split` を維持する。
train 内の minibatch shuffle は学習乱数でよく、split 所属は変えない。

top-level `seed` は model/training 用。`dataset.dataset_seed` は独立して保持し、
今回の deterministic な前処理・split では使わない。
`run.py` が constructor に渡す training RNG を finance loader は明示的に無視する。
比率は build の `--splits` と dataset config、将来の experiment の `splits` を揃える。
不一致はエラーにして暗黙の再分割を防ぐ。PCA、中心点、スケーリング、kernel bandwidth
を今後 fit する場合も train のみで決定する。

## 現在の学習・生成実装

実装順序は Pure Varadhan → ISM → Malliavin。共有の trainer、Haiku network、DSM/ISM loss、
optimizer/EMA、GRW predictor/sampler を使用し、SPD marker のある場合だけ専用処理へ分岐する。
既存 Earthquake / S2 / SO(3) の時間グリッド、乱数生成の既定値、teacher は維持した。

|役割|実装|
|---|---|
|SPD geometry / frame / divergence|`riemannian_score_sde/spd.py`|
|Brownian / finite-time terminal law|`riemannian_score_sde/spd_sde.py`|
|score head|`riemannian_score_sde/models/vector_field.py:SPDGenerator`|
|Malliavin teacher|`riemannian_score_sde/spd_teacher.py:SPDMalliavinTeacher`|
|生成・SPD diagnostics|`riemannian_score_sde/spd_generation.py`|
|評価・可視化|`scripts/evaluate_spd_finance_generation.py`|
|サーバー用テスト|`tests/test_spd_training.py`|

既存ファイルの変更は `main.py`（SPDのみx64）、`run.py`（train split bind・生成呼出し）、
`riemannian_score_sde/losses.py`（SPD ISMのprobe形状）、
`riemannian_score_sde/models/vector_field.py`（SPDGenerator追加）、
`riemannian_score_sde/sampling.py` / `score_sde/sampling.py`（SPD GRW分岐）、
`riemannian_score_sde/teachers.py`（noise次元のoptional引数）、
`score_sde/models/flow.py`（SPD divergence/独立生成乱数）、README とこの文書。
新規configの6ファイルは後述。既存 dataset / preprocessing と他実験configは変更していない。

### AIRM と表現

`AffineSPD` は同梱 `SPDMatrices` の adapter。metric は `SPDMetricAffine` を継承し、
**Affine-Invariant Riemannian Metric を変更していない**。

- state: `(batch,5,5)` の SPD 行列。
- tangent: `(batch,5,5)` の対称行列。
- coordinate: `q=svec(X)` の15成分（非対角は sqrt(2) 倍）。これは ISM divergence 用の座標。
- network input: Cholesky 因子 `L` の log diagonal 5成分と `L_ij/L_ii` (i>j) の10成分。
  全域で滑らかな15特徴を使い、入力側の固有ベクトル微分を避ける。
- network output: 移動正規直交 frame の15係数 → 対称 tangent 行列。
  既存 score/reverse SDE API の境界でのみ25成分に flatten する。

AIRM は

```
g_X(U,V) = tr(X^-1 U X^-1 V)
d(A,B) = ||log(A^-1/2 B A^-1/2)||_F
Exp_X(U) = X^1/2 exp(X^-1/2 U X^-1/2) X^1/2
Log_X(Y) = X^1/2 log(X^-1/2 Y X^-1/2) X^1/2
```

exp の値・log・norm・distance は geomstats の既存実装を使用。
同梱 `SPDMatrices.logm` は正値チェックに Python `if gs.any(...)` を含み JIT 内では使えないため、
`JitSPDMetricAffine._aux_log` は同じ eigenvalue-function 実装を `check_positive=False` で呼ぶ。
正値検査は dataset / artifact 境界で実行し、不正値を補正して隠さない。

Malliavin は exp の高階微分を必要とする。固有値が重複すると naive な eigenvector 微分が
不安定なため、geomstats exp に `custom_jvp` を設け、等価な
`L exp(L^-1 U L^-T) L^T` の Cholesky + JAX Padé expm から JVP を計算する。
これは別 metric や別 forward process ではない。primal は geomstats の値を使用し、
データや固有値に epsilon を足さない。等価性・重複固有値での一次/二次微分はサーバーテスト対象。

### AIRM frame と Brownian GRW

Frobenius 正規直交対称基底を `E_a` とする。
対角基底は `E_ii`、非対角は `(E_ij+E_ji)/sqrt(2)`。
`X=L L^T` の **lower Cholesky** により

```
V_a(X) = L E_a L^T
< V_a, V_b >_X = tr(E_a E_b) = delta_ab
U(X,z) = sum_a z_a V_a(X),   z ~ N(0,I_15)
```

を構成する。`X^1/2 E_a X^1/2` 型の frame と直交変換で結ばれ、同じ AIRM 等方 Gaussian を作る。
Cholesky 版は repeated eigenvalues でも滑らかで、その frame 固有の divergence を明示できる。

`SPDBrownian` は `Brownian` を継承。`tau(t)=integral_0^t beta(s) ds`、区間は `[0,1]`。
各 step は

```
X_{k+1} = Exp_Xk(sqrt(tau(t_{k+1})-tau(t_k)) U(X_k,z_k))
```

既存 GRW predictor に SPD 分岐を追加し、共通 sampler では SPD だけ N+1境界の左端を使う。
forward は0から開始し、eps 分の時間を欠落させない。noise の beta 積分は正確に差分を取る。
これは連続時間 generator `beta(t) Delta_g/2` の geodesic random walk 近似。
`sum V_a circle dB_a` を無補正の Stratonovich SDE とみなしているわけではない。
connection drift は geodesic exp の二次項で反映される。

reverse は既存 `RSDE` の intrinsic drift `-beta(t) score` と同じ tangent Gaussian を使い、
負の時間 step で exp 更新する。reverse drift は Euler 一次近似。終端は `t=eps`。
既定 forward 16 steps / reverse 64 steps は最初の設定であり、離散化収束を確認済みの値ではない。

### 有限時刻の初期分布（非コンパクト性への対応）

SPD には正規化可能な AIRM 一様分布がない。`EmpiricalForwardTerminal` は

```
q_T^GRW = (1/N_train) sum_i Q_T^GRW(. | X_i_train)
```

から、新しい train index と新しい forward noise で生成初期点をサンプルする。
`run.py` で保存済み chronological train split **だけ**を bind する。
reverse noise は terminal sampling と独立の PRNG key を使う。

この方式は **学習データを必要とする有限時刻の empirical terminal sampler** であり、
データ非依存の Gaussian prior、uniform prior、定常分布ではない。
生成時にも train split を参照する点は実験結果に明記する。
3手法とも同じ定義を使い、val/test を prior fitting や初期点抽出へ混ぜない。
将来、データ非依存 prior に置換するときは forward terminal との整合性を別途検証する。

terminal density の解析評価は実装していないので、compact-manifold の log-likelihood / map plot
は config で無効にする。生成後の sample-based evaluation を使用する。

### Pure Varadhan

既存 `VaradhanTeacher` と `get_dsm_loss_fn` をそのまま使用する。
`Langevin.varhadan_exp` の `Log_Yt(Y0)/tau(t)` が AIRM log に委譲する。
Heat teacher や spectral heat kernel への切替はない。
`SPDBrownian.marginal_prob` の sqrt(tau) は network score の preconditioning のためであり、
SPD の厳密な Gaussian standard deviation を主張しない。

既定の DSM weighting は `like_w=true`（beta weighting）、追加の
`exp(-time_weight_lambda*t)` は `loss.time_weighting=true` のときのみ適用。
Pure Varadhan は連続 heat kernel の短時間近似で、有限 GRW の厳密 score ではない。

### ISM divergence

15次元 `q=svec(X)` の Riemannian volume は

```
rho(q) = sqrt(det g(q)) = C det(X)^(-(n+1)/2)
div_g S = tr(D_q svec(S)) - (n+1)/2 tr(X^-1 S)
```

SPD(5) の補正係数は **-3**。`spd.riemannian_divergence` が JAX linearize/JVP で座標 trace を
計算し、解析的な volume term を加える。`get_riemannian_div_fn` は SPD のときだけこの実装へ委譲。
ISM loss は15成分の Gaussian / Rademacher probe を抽出する。
`loss.hutchinson_type=None` は既存 convention の文字列 `"None"` を使う exact trace。
exact と Hutchinson の両経路を用意している。

`<score,score>_X` は既存 metric の squared_norm を使用し、
`0.5 ||score||_X^2 + div_g score` を既存 ISM loss で最小化する。
Euclidean divergence や25個の独立自由度の仮定へ置換していない。

### Malliavin endpoint Jacobian / covariance / correction

`SPDMalliavinTeacher` は既存 `MalliavinTeacher` の noise-space exact/Hutchinson divergence を再利用。
S2 の `-2x` や SO(3) のゼロ divergence をコピーしない。

flatten した noise `Z` は15K次元、endpoint `F(Z)` は5×5。
既存 `compute_endpoint_jacobian` (`jax.jacrev`) で `D_Z F` を計算する。
これは initial state に関する flow Jacobian ではない。

```
J[:,k] = svec(L^-1 (D_Zk F) L^-T),   F = L L^T
Gamma = J J^T                       # 15x15
ridge = covariance_regularization * tr(Gamma)/15
U = J^T (Gamma + ridge I)^-1         # (15K)x15
```

`explicit_grw_endpoint` は forward sampler と同じ beta interval・15次元 noise・exp を使う。
既存 noise helper は次元と dtype を optional に一般化し、既定値3の S2/SO(3) の抽出を維持。

Cholesky frame の field divergence は、**i=0,...,n-1** として

```
div_g V_ii = (n-1-2i)/2
その他（非対角基底）の div_g V_a = 0
```

導出: identity で `D chol(I)[H]` は下三角H、対角だけ1/2倍。
`V_ii` の Euclidean divergence は `n-i`、volume correction は `-(n+1)/2`。
lower-triangular congruence は AIRM isometry であり frame を transport するため全Xで同じ値。
SPD(5) では対角5基底に順に `(2,1,0,-1,-2)` が対応する。

```
delta(U_a) = Z^T U_a - div_Z U_a
score_target = sum_a [-delta(U_a) - div_g V_a] V_a(F)
```

符号は `E[V_a f(F)] = E[f(F)delta(U_a)]`
および integration by parts `= -E[f(F)(<score,V_a>+div_g V_a)]` から得る。
`div_Z U_a` は既存 Hutchinson JVP（既定1 probe）で推定する。
`teacher.divergence_mode=exact` による小規模比較も可能。
endpoint/target は label として stop_gradient する。

ridge=0 かつ J が full rank なら離散 endpoint law に対する IBP identity を使用できる。
既定 relative ridge `1e-8` は inverse を安定化するが target に bias を導入する。
これは **Malliavin covariance** の ridge であり、金融 covariance / SPD state の正則化ではない。
finite K、finite probe、ridge、Pure Varadhan の近似の影響を混同しない。
SPD の Rao–Blackwell smoothing は未実装で、要求されたらエラーにする。

### SPD 維持と precision

forward/reverse とも対称 tangent の exp 更新のみ。各 step の symmetrization は丸め誤差を除くため。
毎stepの eps I、eigenvalue floor、clipping は入れていない。
`enable_x64=true` により `main.py` が JAX import 前に x64 を設定する。

`spd_summary` は非有限値、非対称性、非正固有値をエラーとし、min eigenvalue、determinant、
logdet、condition number を報告する。生成は batch ごとと保存前に検査する。
forward の全履歴についてはサーバーテストで同じ検査をする。
数値不安定で failure になる場合は step数・beta・score・dtype を調べ、データを黙って修復しない。

## Experiment YAML と生成の操作

- `config/experiment/spd_finance_varadhan.yaml`
- `config/experiment/spd_finance_ism.yaml`
- `config/experiment/spd_finance_malliavin_hutchinson.yaml`
- `config/manifold/spd5.yaml`
- `config/teacher/spd_malliavin_hutchinson.yaml`
- `config/beta_schedule/spd_finance.yaml`

beta schedule は専用groupを override して beta_0=0.01 / beta_f=1.0 とする。
`main.yaml` が experiment より後で schedule をロードするため、experiment 内の inline 値だけに依存しない。

既存 `config/dataset/spd_finance.yaml` を再利用する。
`seed` はモデル/学習乱数、`dataset.dataset_seed` は独立した固定データ側の設定。
`generation.seed=123` は3手法で共通、training seed と別。

学習完了時、または `mode=test` で checkpoint を読み込んだ後、`generation.enabled=true` なら
EMA model で `generated_samples.npy` と `generated_samples.metadata.json` を保存する。
既定は256サンプル、batch16、reverse64 steps。`generation.enabled=false` で後回しにもできる。
metadata に terminal law、forward/reverse steps、checkpoint step、各seed、loss、
train split の内容SHA256とSPD検査を保存する。

repo が固定する Hydra 1.1 の cwd=run directory を使用する。
`hydra.runtime.output_dir` は1.1には存在しないため利用しない。
Hydra 1.2+ を使う環境では `hydra.job.chdir=true` を追加して同じ挙動にする。
`mode=test` の再生成では同じ run directory、同じ dataset、同じ architecture/flow 設定を指定する。
1.1 のコマンドへ `hydra.job.chdir` を追加すると未知キーになるので追加しない。

## 評価と可視化

```bash
GEOMSTATS_BACKEND=jax JAX_ENABLE_X64=1 python scripts/evaluate_spd_finance_generation.py \
  --run-dir results/spd_finance_varadhan_seed0 --split test --sigma 1.0
```

移動したデータは `--dataset data/spd_finance/spd_finance_5asset_60d.npz` で指定可能。
`--samples-path`、`--output-dir`、`--chunk-size`、`--max-samples`、`--seed` も指定可能。
`--no-plots` は geometry/finance metrics のみ。
評価は自動で training を開始しない。

- generated / reference 総数と実際の metric subsample 数。
- geomstats の **AIRM distance** による RBF discrepancy（unbiased squared MMD 形式）。
- generated→reference / reference→generated NN の mean/median/std/min/max。
- 固有値、SPD diagnostics、資産別variance、全10ペアcorrelation、最大固有値、trace、det、logdet、condition。
- `log_eigenvalues.png`, `asset_variances.png`, `pairwise_correlations.png`, `matrix_statistics.png`。
- train-only AIRM Fréchet mean を基点とする log-map を Cholesky frame で15座標化。
  train のみで PCA を fit し、同じ中心・basis で real/generated を overlay (`tangent_pca.png`)。
  mean、basis、投影座標は `tangent_pca.npz`。mean の収束状況も metrics に記録。

`k(A,B)=exp(-d_AIRM(A,B)^2/(2 sigma^2))` を使用する。
ただし AIRM geodesic Gaussian は全bandwidthで positive definite を保証できない
([Feragen et al., CVPR 2015](https://arxiv.org/abs/1411.0296))。
`rbf_mmd2_unbiased` は要求された kernel discrepancy として記録し、保証された RKHS distance とは
解釈しない。小さな Gram 行列の最小固有値も記録するが、これは大域的PSDの証明ではない。
unbiased estimator 自体の負値とも区別する。sigma/PCA の選択に test を使わない。
rolling windows が依存するため、信頼区間は将来 block bootstrap 等で検討する。

## サーバーで最初に実行する検証

開発Macでは JAX が AVX binary 非互換で import できず、ユーザーの指定により学習・生成は
実行していない。追加環境のインストールも行わない。
ローカルの確認は Python syntax / YAML parsing / diff check のみ。
**1 update成功、3手法の数値妥当性、実データでの生成成功はまだ主張しない。**

既存GPU環境（vendored geomstats、JAX 0.3.15 系、Hydra 1.1.1 と依存パッケージ）を使う。
PyPIの別バージョン geomstats へ黙って置き換えない。

```bash
GEOMSTATS_BACKEND=jax JAX_ENABLE_X64=1 python -m unittest discover \
  -s tests -p test_spd_training.py -v
```

サーバーテストの内容: frame 正規直交性、15座標の逆変換、geomstats exp/log/distance、
反復固有値での exp 一次/二次微分、explicit/common forward endpoint 一致、forward/reverse SPD、
ISM volume補正、frame divergence、exact/probe trace 比較、SPD(1) lognormal の厳密scoreとの
Malliavin符号照合、3configのHydra composition、**Varadhan→ISM→Malliavin の順で各1 update**、
finite loss/gradient/parameter更新、少数生成のSPD、dataset split と training seed の独立性。

金融データでの end-to-end smoke は下記（順序を守って実行）。共通flagsは
`steps=1 batch_size=2 eval_batch_size=2 flow.N=1 architecture.hidden_shapes='[16,16]'`
`generation.count=2 generation.batch_size=2 generation.steps=2`。
これは計算経路確認用であり、収束精度を確認する設定ではない。

```bash
python -u main.py experiment=spd_finance_varadhan mode=train seed=0 logger=csv \
  steps=1 batch_size=2 eval_batch_size=2 flow.N=1 architecture.hidden_shapes='[16,16]' \
  generation.count=2 generation.batch_size=2 generation.steps=2 \
  hydra.run.dir=results/spd_finance_varadhan_smoke

python -u main.py experiment=spd_finance_ism mode=train seed=0 logger=csv \
  steps=1 batch_size=2 eval_batch_size=2 flow.N=1 architecture.hidden_shapes='[16,16]' \
  generation.count=2 generation.batch_size=2 generation.steps=2 \
  hydra.run.dir=results/spd_finance_ism_smoke

python -u main.py experiment=spd_finance_malliavin_hutchinson mode=train seed=0 logger=csv \
  steps=1 batch_size=2 eval_batch_size=2 flow.N=1 architecture.hidden_shapes='[16,16]' \
  loss.time_weighting=true loss.time_weight_lambda=5.0 \
  generation.count=2 generation.batch_size=2 generation.steps=2 \
  hydra.run.dir=results/spd_finance_malliavin_smoke
```

## GPU server 本番コマンド（未実行）

まず上記の検証を通してから実行する。100kの学習品質や計算量を確認済みという意味ではない。
Malliavinは15K noise の endpoint Jacobianとその微分を使うため、特にcompile時間・GPU memory・
1 update所要時間を確認する。必要なら batch_size / flow.N を調整し、3手法で条件を揃える。

```bash
python -u main.py \
  experiment=spd_finance_varadhan \
  mode=train seed=0 steps=100000 logger=csv \
  hydra.run.dir=results/spd_finance_varadhan_seed0

python -u main.py \
  experiment=spd_finance_ism \
  mode=train seed=0 steps=100000 logger=csv \
  hydra.run.dir=results/spd_finance_ism_seed0

python -u main.py \
  experiment=spd_finance_malliavin_hutchinson \
  mode=train seed=0 steps=100000 logger=csv \
  loss.time_weighting=true loss.time_weight_lambda=5.0 \
  hydra.run.dir=results/spd_finance_malliavin_lambda5_seed0
```

lambda別 YAML は作成せず、同一configへ `loss.time_weight_lambda=0.0/5.0` を override する。
各runの学習終了時に生成される。別途再生成する例（本番と同じ architecture/flow を維持）：

```bash
python -u main.py experiment=spd_finance_varadhan mode=test seed=0 logger=csv \
  hydra.run.dir=results/spd_finance_varadhan_seed0 \
  generation.count=1000 generation.batch_size=16 generation.steps=64
```

## データ再生成（既存の前処理）

```bash
python scripts/build_spd_finance_dataset.py --download --include-jgb-candidate \
  --start 2008-01-01 --end 2026-09-08
python scripts/build_spd_finance_dataset.py --start 2008-01-01 --end 2026-09-08
python -m unittest discover -s tests -p test_spd_finance_preprocessing.py -v
```

今回 raw data / NPZ / split / regularization は変更していない。
既存前処理の6テストと全1801行の独立再計算は前回確認済み。

## 未解決・近似の範囲

- サーバーでの実行確認は未実施。Hydra composition と JAX の数値テストもサーバーで確認する。
- GRW と reverse Euler は有限step近似。continuous-time convergence study は未実施。
- Pure Varadhan は短時間近似。Malliavin は離散endpoint law、relative ridgeにはbias、Hutchinsonには分散がある。
- 生成初期分布は train empirical forward law。データ非依存 prior からの生成や likelihood は未対応。
- Fréchet mean iteration は有限反復。収束フラグを確認し、未収束の場合は「近似共通基点のPCA」と解釈する。
- 小規模 Gram 検査は AIRM Gaussian kernel の大域的 positive definiteness を保証しない。
- 実金融データでのモデル品質、regime generalization、長時間学習、100k/600k、本評価は未実施。
