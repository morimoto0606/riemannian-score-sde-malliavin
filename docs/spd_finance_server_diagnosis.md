# SPD finance サーバー実行障害の診断

最新状況：サーバーでconfigテスト2件とCPU tiny smoke（3手法、各1 update・8件生成）が成功。
詳細は [CPU smoke検証結果](spd_finance_cpu_smoke_result.md) を参照。
以下の未実行・障害記述は診断時点の記録であり、GPUと本番shapeの検証は引き続き未完了。

## 結論と証拠

現在は独立した二つの実行障害がある。金融データを使うsmokeはNPZの欠落で停止し、
データを使わない単位行列のGPU CholeskyテストはcuSolver共有ライブラリをロードできず停止する。
この二つを修復する前にSPDの数学実装を変更しても、提示された停止原因は解消しない。[1][2]

| 対象 | 観測 | 判断 |
|---|---|---|
| 金融smoke | `data/spd_finance/spd_finance_5asset_60d.npz` がない | データ配置が必要 |
| 最小GPUテスト | `libcusolver.so.11` をロードできない | ライブラリの欠落または検索経路の不整合 |
| コンパイラ | `ptxas 11.0.221`、11.1未満の誤コンパイル警告 | cuSolverとは別に対処が必要 |
| GPU認識 | JAXが `GpuDevice(id=0)` を表示 | デバイス列挙まで成功。行列分解の成功は意味しない |
| GPU負荷 | V100、15170/32510 MiB、98%、Python 6プロセス | 高負荷だが、今回の直接原因をOOMと断定できない |
| ドライバ | 450.51.06 | 新しいToolkitへの切替は互換性確認が必要 |

メモリ事前確保を無効にした最小テストでも共有ライブラリのロード失敗が出ている。
したがって「他のPythonを停止すれば直る」という推定には根拠がない。表示されたプロセスは
停止・リセットの対象とせず、スケジューラで割り当てられた実行環境の整合性を調べる。[1][2]

## データの復旧

学習用NPZ（約447 KiB）と対応するmetadata JSONはGit管理対象に変更した。
raw downloadは引き続き除外する。診断時点ではディレクトリ全体が除外されていたため、
サーバーにsnapshotが届いていなかった。今後はこの2ファイルを含むコミットをpushした後、
サーバーで同じブランチをpullすれば配置できる。データ再取得・再計算は不要である。[3]

| ファイル | SHA256 |
|---|---|
| `spd_finance_5asset_60d.npz` | `e7944eda5d9b22a647fa323ce7bb40cf9e292a057d7bc7d228047f2d88db531b` |
| `spd_finance_5asset_60d.metadata.json` | `559285c763305fd63b9b89f63c489f5d62d5d7cf226ddb724bb4cdbd2c999a8a` |

```bash
cd ~/riemannian-score-sde-malliavin
git pull --ff-only
```

サーバーのリポジトリ直下でハッシュと内容を確認する。NPZ内のmetadataとsplitを読み、
JAXやGPUは使用しない。

```bash
sha256sum data/spd_finance/spd_finance_5asset_60d.npz \
          data/spd_finance/spd_finance_5asset_60d.metadata.json
python - <<'PY'
import numpy as np
with np.load('data/spd_finance/spd_finance_5asset_60d.npz', allow_pickle=False) as d:
    x = d['covariances']
    assert x.shape == (1801, 5, 5)
    assert np.isfinite(x).all()
    ev = np.linalg.eigvalsh(x)
    counts = [len(d[k + '_indices']) for k in ('train', 'val', 'test')]
    assert counts == [1260, 210, 211]
    assert np.all(ev > 0)
    print('shape:', x.shape, 'splits:', counts)
    print('symmetry error:', np.max(np.abs(x - x.swapaxes(-1, -2))))
    print('minimum eigenvalue:', ev.min(), 'non-SPD count:', np.sum(ev[:, 0] <= 0))
PY
```

別の共有ディスクに同じsnapshotが既にあるなら、転送せず以下でもよい。

```bash
python scripts/smoke_spd_finance.py --dataset /absolute/path/spd_finance_5asset_60d.npz
```

今回のスクリプト修正では、NPZがない場合はresultsディレクトリの作成や学習プロセスの起動より前に
明確なエラーで終了する。`--dataset` は現在の3手法すべてのHydra設定へ絶対パスとして渡される。
データを自動ダウンロードしたり、代替データを生成したりする動作はない。

## CUDA環境の診断

`nvidia-smi` のCUDA表示は、インストール済みToolkit一式の証明ではない。
NVIDIAはこの表示をドライバがサポートするCUDAバージョンとして説明している。
Toolkitの実体、実行されるptxas、Pythonプロセスがロードする共有ライブラリを分けて確認する。[4]

CUDA 11.1の公式リリース表ではcuSolverコンポーネントは11.0.0.74である。
CUDA 11以降はToolkit各コンポーネントの版が独立しているため、CUDAという名前に11が
含まれているだけでは必要なcuSolver ABIが存在するとは限らない。[5]

割り当て済みGPUノード上で以下を採取する。環境の変更やインストールは行わない。

```bash
hostname
nvidia-smi
command -v python
python -m pip show jax jaxlib
type -a ptxas
ptxas --version
printf 'CUDA_HOME=%s\nLD_LIBRARY_PATH=%s\n' "${CUDA_HOME:-}" "${LD_LIBRARY_PATH:-}"
if type module >/dev/null 2>&1; then
  module list 2>&1
  module avail cuda 2>&1
fi
ldconfig -p 2>/dev/null | grep -E 'libcusolver|libcublas|libcudart'
python - <<'PY'
import ctypes
try:
    ctypes.CDLL('libcusolver.so.11')
    print('cuSolver DSO load: PASS (not yet a numerical test)')
except OSError as error:
    print('cuSolver DSO load: FAIL:', error)
PY
```

`ldconfig` にないことだけでは未インストールとは断定しない。moduleが提供する非標準パスもあり、
`ctypes.CDLL` の結果と合わせる。ファイルがあるのにロードできない場合は、その実ファイルの
`ldd /verified/path/libcusolver.so.11` で依存ライブラリの欠落を調べる。
ライブラリ名だけの偽装symlinkはABIの整合性を作らないため、修復方法としない。

## ドライバとコンパイラの整合性

CUDA 11.1 GAの通常のLinuxドライバ要件は455.23以上と記載されている。[5]
一方、CUDA 11.xのminor-version compatibilityには450.80.02への更新という条件と、
PTX等に関する制約が記載されている。今回の450.51.06はこの基準より古く、
「CUDA 11.xならすべてそのまま使える」と判断できない。[6]

したがって推奨する復旧経路は、管理者が確認したJAX 0.3.15用のdriver／Toolkit／cuSolver／
cuDNN／ptxasの組合せを持つノードまたはmoduleを使用することである。実在するmodule名や
互換性パッケージの有無はまだ不明であり、特定の `module load cuda/...` を確定コマンドとして
提示する段階ではない。既存venvを無条件で最新JAXへ更新する必要性も現時点では示されていない。

添付ログはptxas 11.1未満に誤コンパイル・不正アドレスの既知リスクがあることを明示している。
ptxasだけの交換ではcuSolver欠落が残り、cuSolverだけの配置ではこの警告が残る。
両方の問題を別々に解決し、行列分解を再検証する必要がある。[1]

## 再開の順序と合格条件

1. NPZを配置してハッシュ・shape・split・SPD性を確認する。
2. GPU環境の修復を待つ間、CPUの小規模smokeで学習コード経路を検証する。
3. 整合したGPU環境で単位行列Choleskyと固有値分解を通す。
4. 3手法の数理テスト、1 update、checkpoint復元、少数生成を実施する。

CPU実行は新環境を必要としない。JAXのbackend選択でCPUを指定する。[7]

```bash
JAX_PLATFORM_NAME=cpu CUDA_VISIBLE_DEVICES="" \
python scripts/smoke_spd_finance.py --tiny --timeout 600
```

CPU成功はGPU成功でも本番shapeでの検証成功でもない。GPU環境の修復後は以下の最小計算を行う。

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=true python - <<'PY'
import jax
import jax.numpy as jnp
import numpy as np
print(jax.devices())
assert jax.default_backend() == 'gpu'
x = jnp.eye(5, dtype=jnp.float64)
l = jax.jit(jnp.linalg.cholesky)(x).block_until_ready()
w = jax.jit(jnp.linalg.eigvalsh)(x).block_until_ready()
np.testing.assert_allclose(np.asarray(l), np.eye(5))
np.testing.assert_allclose(np.asarray(w), np.ones(5))
print('GPU float64 factorization PASS')
PY
```

最小GPU計算が通った後に既存のserver検証を再開する。新しいsmokeは自動で別ディレクトリを使い、
既存checkpointを再利用して上書きすることはない。現時点で1 update・生成・生成物の非SPD数0を
実測済みとはいえない。GPU共有ライブラリの実配置と対応するmodule情報も未確認である。

## Sources

1. 添付 `pasted-text.txt`（GPU単位行列テスト、2026-09-11 22:59:35–36）。非公開実行ログ。
   JAX/jaxlib 0.3.15、ptxas警告、libcusolverロード失敗、cuSolver例外を記録。
2. サーバー実行ログ（会話内、2026-09-11）：NPZのFileNotFoundErrorと22:58:59のnvidia-smi出力。
   非公開記録。プロセスの所有者・ジョブ種別は未確認。
3. リポジトリ一次資料：`.gitignore`、`config/dataset/spd_finance.yaml`、
   `data/spd_finance/spd_finance_5asset_60d.npz` と付随JSON。ローカルファイルのSHA256を上記に記載。
4. NVIDIA, [NVIDIA System Management Interface — CUDA UMD Version](https://docs.nvidia.com/deploy/nvidia-smi/index.html),
   更新日記載なし、2026-09-11参照。ドライバの対応CUDA表示の意味。
5. NVIDIA, [CUDA Toolkit 11.1.0 Release Notes, Tables 1–2](https://docs.nvidia.com/cuda/archive/11.1.0/cuda-toolkit-release-notes/index.html),
   2020-10-05。cuSolverコンポーネント版とCUDA 11.1の通常ドライバ要件。
6. NVIDIA, [Minor Version Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html),
   更新日記載なし、2026-09-11参照。CUDA 11.xの450.80.02条件、PTX等の制約。
7. JAX, [Configuration Options — Platform Name](https://docs.jax.dev/en/latest/config_options.html),
   更新日記載なし、2026-09-11参照。backend選択。現行文書ではJAX_PLATFORM_NAMEは非推奨だが、
   ここでは既存0.3.15環境向けの従来指定を用いる。
