# R-NaDドラフトのピック閲覧・検証（2026-11-17）

対象: Issue #5 / 既存Draft PR #6。モデルやcheckpoint、旧教師の既定処理は変更しない。

## 3種類の記録を区別する

1. **教師の実際の受理済みピック**：export-teacher.mjs の opt-in --draft-review true で、自席の継続所持・提示候補・ポケモン名・型ID・既知技/特性/持ち物/役割/タイプ/能力・予算・全合法6体候補・AQ公開utilityを保存。
2. **学生による教師到達局面の予測**：report_draft_picks.py は上記トレースと固定モデルから、教師Top5、学生Top5/確率、選んだ6体、教師utility順位とregret、seat/seed/score/source SHAをMarkdownとJSONへ出力。学生が実戦で受理した行動ではない。
3. **学生自身の権威Matchで受理されたピック**：audit-live-draft-picks.mjs は凍結v4ドラフト＋v3選出戦闘を本物AQ-Base-001と完全4勝Matchで対戦させ、commit後に受理された6体・費用・自軍候補と公開AQ評価を記録。1Matchでは強度確証とはしない。

閲覧: [初回教師ピック診断run #43983248201](https://github.com/cueikusuta/p-RL1024/actions/runs/43983248201)。Job summaryのポケモン名・ランキング、Artifact内のDRAFT_PICKS.md / draft-picks.json。Artifact保持30日。

## ドラフト改善1〜4の研究比較

1. **特徴量**：自軍の既知能力値6/タイプ18/役割タグ17/費用・Mega・carryの計44次元。6体チームの合計費用・残り予算・carry数・Mega数の4次元。相手未公開型は使わない。
2. **専用構造**：加算型Linearと、各ポケモンを共有MLPでエンコードし平均・最大・2次モーメントを対称集約するDeep Setsを、同じ教師dataで比較。順序不変性はテスト。
3. **教師順位の蒸留**：hard Top1 CEと、公開AQ utilityのTop12を温度付き確率にしたsoftランキング損失を比較。Top1/Top10/正規化CE/正規化utility regretを計測。
4. **教師データの増量**：同じデータ系列の4対16完全4勝Matchを、別の4完全Match heldoutで比較し、入力・学習条件・SHAを保全。

実装: scripts/rnad/train_draft_deepsets.py / test/rnad_draft_deepsets_test.py。
Workflow: .github/workflows/rnad-i5-draft-deepsets-study.yml。
実験条件: docs/research/rnad/DRAFT_DEEPSETS_STUDY_MANIFEST.json。

注意: Deep Setsの重みは現時点でゲームの実行経路に接続していない研究用モデル。模倣学習結果をそのまま対AQの強度改善とは言わない。実装した後、別の未使用seed×両席の完全4勝Match、対応ありΔ/信頼区間、最新HEAD CI、実装担当とは別ChatGPT Chatの独立レビューがChampion昇格条件。

初回の既存v4教師模倣の参考値：ドラフトTop1 6.52%、Top10 19.02%、正規化CE 0.937。これらと新実験の条件差は比較時に明示する。
