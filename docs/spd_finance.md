# 金融実データの SPD(5) 実験準備

2026-09-08 に取得・前処理を実行。学習、本評価、生成サンプリングは未実行。
指定された `/home/export/home/ymorimoto/github/riemannian-score-sde-malliavin`
はこの環境にないため、ワークスペースの同名リポジトリに実装した。
データ準備は完了。SPD の学習 pipeline は未対応箇所があるので、実行可能と
誤解される `spd_varadhan.yaml` / `spd_ism.yaml` /
`spd_malliavin_hutchinson.yaml` は作成していない。

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

## 同梱 geomstats と pipeline の適合性

この repository の API 名で記述する。新しい geomstats の `equip_with_metric` 等と混同しない。

|項目|既存実装・判定|
|---|---|
|manifold|`geomstats/geomstats/geometry/spd_matrices.py: SPDMatrices(n=5)`|
|metric|デフォルト `SPDMetricAffine(n=5, power_affine=1)`。Affine-Invariant metric|
|exp / log|`space.metric.exp(U,X)`, `space.metric.log(Y,X)`。`space.exp/log` も metric へ委譲|
|distance|`space.metric.dist(A,B)` を基底クラスから継承。`norm(log(B,A),A)` による距離|
|tangent|5×5対称行列、内在15次元。`OpenSet.to_tangent` → `SymmetricMatrices.projection`|
|random point|`SPDMatrices.random_point` は有界対称行列の指数。非コンパクト空間の一様分布ではない|
|projection|対称化＋固有値を `gs.atol` へ floor。今回不使用|
|Gaussian tangent|継承 `OpenSet.random_normal_tangent` は `(batch,15)` を返す。AIRM の5×5 Gaussian tangent は未実装|

距離は `||log(A^(-1/2) B A^(-1/2))||_F`。
`g_X(U,V)=tr(X^(-1) U X^(-1) V)`。
`Exp_X(U)=X^(1/2) exp(X^(-1/2) U X^(-1/2)) X^(1/2)`、
`Log_X(Y)=X^(1/2) log(X^(-1/2) Y X^(-1/2)) X^(1/2)`。
新規 manifold は追加していない。

### Varadhan

数学的には利用可能。`VaradhanTeacher.score_at_endpoint` →
`Langevin.varhadan_exp` が既に `manifold.log(y0,yt)/tau` を使い、
tau を `(batch,1,1)` に拡張するので行列 endpoint の target 部分は流用できる。
`tau=beta_schedule.rescale_t(t)-rescale_t(0)`。
ただしこの target は短時間近似で、SPD の厳密な有限時間 heat-kernel score ではない。

**teacher 全体・training の無変更流用は不可。** `sample_and_score` の
forward sampler が SPD の正しい Brownian motion を生成できない。
GRW predictor は tangent noise を flatten して25成分を要求する一方、現実装は15成分。
対称 Frobenius 正規直交基底 `E_a`（非対角は `1/sqrt(2)`）を使い
`U = X^(1/2) (sum_a z_a E_a) X^(1/2)` とする AIRM 等方 Gaussian が必要。
その geodesic random walk と beta/time 規約を検証する必要がある。

`Brownian` の terminal/base は compact manifold 用 `UniformDistribution`。
SPD(5) は非コンパクトで、正規化された AIRM 一様分布はない。
単に `random_point` を prior に置換しても forward terminal 分布とは一致しない。
confining Langevin と適切な参照分布、または有限時刻 terminal approximation などの設計が必要。
既存 `WrapNormDistribution` も identity、normal tangent、logdetexp 等を仮定し、
SPD へ差し替えるだけでは成立しない。Langevin に変更すれば有限時間 teacher の意味も再検討する。

### ISM

**そのまま使えない。** `riemannian_score_sde/losses.py:get_ism_loss_fn` の
metric squared norm は利用できるが、`score_sde/models/flow.py:get_riemannian_div_fn`
は `metric.lambda_x` がなければ体積係数1とする。SPDMetricAffine にはこの属性がなく、
AIRM の体積要素を落とす。`score_sde/utils/jax.py:get_estimate_div_fn` も flatten した
noise と model output の shape 合意を要する。

独立対称成分の座標 q では `sqrt(det g) ∝ det(X)^(-(n+1)/2)`。
`div_g f = (1/sqrt(det g)) sum_a partial_a(sqrt(det g) f^a)` を15座標で実装し、
Hutchinson も独立成分・metric の規約に合わせる必要がある。
25要素を独立な自由度として扱った ambient divergence を無条件に使わない。
`AmbientGenerator` は15出力をそのまま tangent projection に渡し、
`LieAlgebraGenerator` は Lie group と `dim×dim` reshape を仮定するため、どちらも無変更では不可。
15係数→正規直交対称基底→5×5 tangent を明示する score head が必要。

### Malliavin Hutchinson

**そのまま使えない。** `MalliavinTeacher._manifold_kind` が S2 / matrix SO(3) 以外を拒否。
`sample_upstream_grw_standard_noise` も毎ステップ3次元。

必要な変更は、(1) 15次元 noise を使う SPD AIRM GRW endpoint、
(2) `V_a(X)=X^(1/2)E_aX^(1/2)` 等の driving fields と generator 規約の確認、
(3) frame とその Riemannian volume divergence、(4) noise derivative `D_Z X_t` を
25 ambient 成分から15 tangent 係数へ移す metric-aware 射影、(5) 15×15 Malliavin covariance、
(6) Skorokhod weight `Z^T U - div_Z U` と endpoint-field divergence correction の導出。
SO(3) のゼロ divergence や S2 の `-2x` を移植しない。

`D_Z X_t` は shape `(25,15*K)`、tangent Jacobian は例えば各列を
`svec(X^(-1/2) (D_Z X_t) X^(-1/2))` にして構成する。
それから `C=J J^T` を作る。ユークリッド `frame.T @ flattened_J` をそのまま使うと
AIRM 射影にはならない。generic な covariance / noise-space Hutchinson JVP 部分は流用候補。
frame 再構成、符号、補正項は導出・exact divergence との小規模比較が必要。
endpoint Jacobian の JAX 微分、固有値重複付近の sqrt/log/exp の高階微分と
条件数も検証する。今回、この大規模拡張は行っていない。

## 評価設計

AIRM `metric.dist` の pairwise adapter で generated→reference NN と
reference→generated NN を計算可能。距離計算は chunk 化する。
例えば対応環境では `metric.dist(A[:,None,:,:], B[None,:,:,:])` の shape と値を
先に数行で確認する。既存の未追跡 `scripts/generation_metrics.py` にも距離関数を
受け取る汎用処理があるが、今回の成果物の依存にはしていない。

`exp(-d_AIRM^2/(2*sigma^2))` の RBF 統計量自体は計算できるが、AIRM SPD では
すべての bandwidth に対する positive-definite kernel 保証がない。
[Feragen et al., CVPR 2015](https://arxiv.org/abs/1411.0296) の結果を踏まえ、
自動的に RKHS-MMD と解釈しない。候補 sigma と有限 Gram 行列の固有値を確認する。
保証された比較としては、補助評価に `svec(log X)` 上の通常の Euclidean Gaussian MMD を併記できる。
これは AIRM geodesic MMD と同一ではない。unbiased MMD 推定量の負値と
kernel が非正定値である問題も区別する。bandwidth は train/reference で固定する。

可視化候補: 昇順 log eigenvalues 5成分、log determinant、trace、condition number、
train のみで求めた AIRM Fréchet mean での log-map 座標の PCA / UMAP。
PCA では `Xbar^(-1/2) Log_Xbar(X) Xbar^(-1/2)` を対称正規直交基底の15成分にし、
fit は train のみ。val/test の時間区間も色分けし、市場 regime 差を確認する。
重なる rolling windows のため IID bootstrap ではなく時系列 block bootstrap を検討する。
本評価・可視化の本格実装は今回行っていない。

## 再生成・検証コマンド

repo root で、NumPy/Pandas を利用できる Python を使う。この Mac では
`../scoremodel/.venv-riemannian/bin/python` で前処理・単体テストを実行できた。
新しい環境なら `python -m pip install -r requirements_spd_finance.txt`。

```bash
# 最初の取得と build（6要求のみ、候補2561も調査）
python scripts/build_spd_finance_dataset.py --download --include-jgb-candidate \
  --start 2008-01-01 --end 2026-09-08

# 同一 raw snapshot から再生成、通信なし
python scripts/build_spd_finance_dataset.py --start 2008-01-01 --end 2026-09-08

# 小規模・offline の検証
python -m unittest discover -s tests -p test_spd_finance_preprocessing.py -v
```

最新日まで更新するときは `--download` を付け、`--end` を翌日ではなく今日の日付にする
（終了日は exclusive）。snapshot を保存したい場合は別の `--output-dir` を指定する。
既定 output-dir の再取得は既存 snapshot を更新する。

6件の offline テスト通過。共分散定義、欠損を跨がないこと、未来の価格変更が過去窓を
変えないこと、split の価格非共有、正則化の明示性、隔離規則と live bar 除外を検証。
さらに実 NPZ の全1801行を `np.cov(ddof=1)` で独立再計算し一致、FX の観測日、
train/val/test の境界非共有を検査した。

環境制約: この Mac の既存 Python は x86版 jaxlib の AVX 要件に合わず JAX import 不可。
同梱 geomstats の NumPy backend も `autodiff.grad` の API 欠落で import 不可。
したがって geomstats / Varadhan / loader / run.py の JAX 実行 smoke は未実施であり、
上記の pipeline 判定はローカルソースの調査結果。
対応する Linux/JAX 環境で、まず import・exp/log inverse・距離対称性・微分・loader split を
数行だけ確認する必要がある。前処理の NumPy 検証と学習の動作確認は区別する。

## 学習前の作業と将来の command 案

必要な順序は、対応 JAX 環境と float64、AIRM tangent noise / forward process / terminal prior、
15係数の score head と形状変換、metric-aware divergence、Varadhan の最小 loss/gradient
smoke、必要に応じて Malliavin の数学・実装拡張、そして3種類の experiment YAML 作成。
PSD/kernel 評価設計、sampler と terminal 分布の整合性も学習前に決定する。
`SPDFinanceDataset` が `TensorDataset` で JAX array 化されるので、float64 を維持するには
JAX import 前に `JAX_ENABLE_X64=1` を設定する。

下記は **追加実装・YAML 作成後だけ有効となる案**。現在は存在しない experiment を指定するため
実行不可。steps は本学習の推奨値ではなく、最初の1更新 smoke の想定である。

```bash
GEOMSTATS_BACKEND=jax JAX_ENABLE_X64=1 python main.py \
  experiment=spd_varadhan seed=0 dataset.dataset_seed=0 \
  data_dir=./data splits='[0.7,0.15,0.15]' \
  mode=train steps=1 batch_size=2 eval_batch_size=2 \
  train_val=false train_plot=false test_val=false test_test=false test_plot=false
```

対応後は `experiment=spd_ism` / `spd_malliavin_hutchinson` へ変更し、同一 snapshot・
split を維持して `seed` のみ変える。100k / 600k や本評価の command はまだ確定しない。
