# R-NaD Issue #5：ドラフト強度確証・固定N評価（2026-11-17）

対象: [Issue #5](https://github.com/cueikusuta/p-RL1024/issues/5) / [Draft PR #6](https://github.com/cueikusuta/p-RL1024/pull/6)。

**目的**：今までの4～32独立seedの探索結果からモデルの優劣を判断しない。候補モデルを一つだけ固定し、事前に必要規模を宣言した完全4勝Matchを全数終了してから、対応あり勝率差とAQ-Baseに対する絶対勝率を判定する。

## 事前登録済みの確認条件

正本：[DRAFT_CONFIRMATION_PLAN_2026-11-17.json](./DRAFT_CONFIRMATION_PLAN_2026-11-17.json)。

- 候補：**Deep Sets・教師16完全Match版**（`draft-deepsets-soft`）、モデルSHA256 `f3d19e30a5b469582309a22331d8830362f681cc6bd5e16e6b5b8608fa8b918f`。この候補は過去の**探索データを見て選んでいる**。従って過去の探索runは確証標本に混ぜない。
- 基準：凍結v3（`all-v3`）、SHA256 `c5932531a2ed9e54079dbd915962624dadabc28a2e67893f0a38e96b4adeb75b`。両モデルとも選出・戦闘は同じ凍結v3。
- 相手：本物の凍結 **AQ-Base-001**、SHA256 `5ccc11a3362c7f5a8874847e3afc1f19c539d4ef258bd82704f53d8a1db1946e`、固定エンジンコミット `ea81165993110ee9fb3a09dcb97c6708c0d7397b`。
- seed：プラン内の固定ネームスペースと連番をSHA-256変換して**新規3,072個**生成。重複・既知探索seedとの衝突を拒否。seed 1個につき**2モデル×2席の4完全Match**。合計 **12,288完全Match**、実験計画を48シャード（64seedずつ）に分割。
- 実行：GitHub-hosted `ubuntu-latest`、`workflow_dispatch` **手動のみ**、`max-parallel: 2`、`fail-fast: false`、一つの実験全体でグローバルな重複実行を抑止。ユーザーが一度開始すると内部48シャードを自動処理する。
- リトライ：失敗したシャードだけを同一Runで再実行可能。成功済みシャードのArtifactを再利用。途中で欠損が残った場合は集約せず、**未結論**とする。
- モデル重みは元Run（v3 #43829072730、v4 #43957735875、Deep Sets #43986137367）から**prepareで一度だけ**取り込みSHA検証後、実行Runの1つのArtifactで全シャードに配布。元Artifactの保存期限が切れている場合は自動的に失敗し、重みの同一性を検証しないまま代理モデルに差し替えない。
- ソースのGit HEADとプランのバイト列SHAは**試験開始前にfreeze**。全シャードが同一Gitコミット、Plan SHA、モデルSHA、元AQとエンジンSHAであることを検査。

## 統計判定（固定Nを最後まで実行する）

各seedの席入替済み結果を

`D_s = (候補の席0勝 + 候補の席1勝 - v3の席0勝 - v3の席1勝) / 2`

で集約し、独立な計数単位を**seed**とする。2席や4モデルの個別Matchを独立標本として水増ししない。

- 第1条件：対応あり `Δ = mean(D_s)` の**信頼区間下限が0を超え**、かつ観測値が**+5ポイント以上**。
- 第2条件：候補の元AQ-Baseに対する両席平均勝率の**信頼区間下限が50%超**。
- CI：seedごとの区間 `D_s∈[-1,1]`／AQ勝率 `W_s∈[0,1]`に適用した**固定標本数の経験Bernstein区間**。2指標×両側の4つの信頼限界それぞれに `α/4=0.0125` を割り当て、Bonferroni補正による**同時95%以上**の区間。元のHoeffdingより分散に応じて鋭いが、**固定N以外や途中停止には適用不可**。
- 第3条件：最新HEADの必要CIと**実装とは別ChatGPT Chatセッションの独立レビュー**成功。統計の合格は独立レビュー依頼の**候補資格**であり、プログラムはChampionを自動昇格せず、PRも自動マージしない。

**検出力**：n=3,072は過去32seedの対応あり差標準偏差≈0.523と真の差+5ptを参考に余裕を持って設定した**設計値**。事前推定誤差や絶対AQ条件を考慮すると、必ず有意になる保証はない。大規模評価が未実行なら「性能不明」であって「有意差なしの確定」ではない。

## 実行入口と記録

- 手動Workflow：[rnad-i5-fixed-n-confirmation.yml](../../../.github/workflows/rnad-i5-fixed-n-confirmation.yml)（GitHub Actionsタブから `Run workflow`）。`workflow_dispatch`だけで、通常のpushで**重い12,288Matchは自動起動しない**。
- プロトコル：`scripts/rnad/draft-confirmation-protocol.mjs`、ジョブ：`scripts/rnad/draft-confirmation-shard.mjs`、集約：`scripts/rnad/draft-confirmation-aggregate.mjs`。
- 必須契約テスト：`test/rnad-draft-confirmation.test.js`（Stage 0 CIへ統合）。
- `i5-confirm-frozen-models`：実験Run内で再配布する検証済み重み。
- `i5-confirm-shard-0` ～ `i5-confirm-shard-47`：個別64seed、各seed全席・受理済みドラフト・提示候補・費用・4勝Match結果。90日保持予定（リポジトリ設定が優先）。
- `i5-confirm-complete-report`：全seed完全性を確認した後の `fixed-n-report.json` / `per-seed.csv` / `CONFIRMATION_REPORT.md`。全部揃わなければ**このArtifactは生成しない**。

**重要**：Workflowのsource/plan SHAは実行Runに固定される。失敗ジョブを再試行するときは同じRunの `rerun-failed-jobs` を使う。別Runで評価結果を見てモデル・seed・試験数を変更してやり直す行為は、新たな探索と見なす。試験内容を更新するなら**新しい事前登録と未使用seed**が必要。

過去の[統計監査](./DRAFT_EVIDENCE_STATISTICAL_AUDIT_2026-11-17.md)に示した小標本の「最良」「改善」はすべて撤回済み。次の結果を表示する前に全3,072seedの完全性とCIを検証する。
