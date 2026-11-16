# R-NaDドラフト・選出：模倣の中間目標と実戦強度ゲート（2026-11-17）

対象：Issue #5、既存Draft PR #6。これは性能達成宣言ではなく、事前に固定する評価契約。

## 最終目的

**AQ-Base-001の凍結モデルを独立seed・両席入替の完全4勝Matchで上回ること**。模倣一致率は目的ではなく、観測と教師信号が効いているかの中間診断。Champion昇格には十分な未使用holdout、paired seed単位の95% CI下限が対AQで50%を超えること、別ChatGPT独立レビューと最新HEAD必須CIが必要。候補評価で用いたseedを最終holdoutへ流用しない。実務上は旧R-NaDから+5ptの改善を狙うが、+5ptは統計的有意差の代用ではない。

## 初期数値の根拠

Run #43812943402の凍結v2モデル／旧aq-public-belief教師の独立8完全Match（各phase88 decisions）では次の通り。Top1だけでフェーズの学習難度を順位付けしない。

| phase | Top1 | Top10 | mean normalized CE | 平均候補数 | 一様Top1 |
|---|---:|---:|---:|---:|---:|
| draft | 3.41% | 17.05% | 0.856 | 1690.69 | 0.10% |
| selection | 4.55% | 46.59% | 0.900 | 60 | 1.67% |

**暫定スクリーニング目標**（合理性の実証値ではなく改善を追跡する工学上の閾値）：
- Draft Top1 8%以上、Top10 30%以上、normalized CE 0.78以下。発展目標Top1 12%、Top10 40%、CE 0.70以下
- Selection Top1 15%以上、Top10 65%以上、normalized CE 0.75以下。発展目標Top1 25%、Top10 80%、CE 0.65以下
- 目標に到達してもAQ勝率改善がない場合は性能成功としない。未達でも実戦勝率が有意に改善すれば採用候補として扱う

draftは約1691候補のため低い絶対Top1でもランダムより高い。初期8%は約2.3倍を目指す診断目標。selectionは候補60と少なく旧教師20・学生60・本物AQ120の構造差があり、教師修正後の改善余地を確認するため15%を設定。ただし**教師バージョン変更だけでTop1差分を学習効果と扱わない**。

## 利用する既存検証器と新規ゲート

1. 既存 `scripts/rnad/evaluate_aq_imitation.py`：未使用受理済み完全4勝Matchで、候補数、Top1/3/10、正規化CE、教師ラベル確率、モデル/teacher/source/engine SHAを報告。既存8Matchは根拠/診断用であって採用判定の標本ではない
2. 新規 `scripts/rnad/phase_imitation_goals.py`：**同一教師ID/特徴schema/コードソース/engine/seed/Match数**で得た旧・新学生レポートを比較。各phase最小256個の教師判断、目標到達かつTop1/Top10/CEが全て改善した場合だけ `goal_screen_pass`。必ず `strength_established=false` と `is_champion=false` を出す。例：
```sh
python -m scripts.rnad.phase_imitation_goals \
  --baseline artifacts/baseline/heldout-imitation.json \
  --candidate artifacts/candidate/heldout-imitation.json \
  --out artifacts/phase-screen.json
```
3. 新規 `scripts/rnad/selection-oracle.mjs`：本物のAQの公開情報ベース選出スコア120行（先発、控え順あり）を権威60合法候補へ対応付け、120行が60候補各2回・控え順によるスコア差ゼロでなければfail-closed。旧`aq-public-belief`教師は不変、新実験`aq-public-strategy-selection`（ID `rnad-aq-public-strategy-selection-v1`）のみ新teacherを使用。実モデル・学習済み重みは変更せず

## 次の実験順

1. **選出教師の受理smoke**：`node scripts/rnad/export-teacher.mjs --seed <未使用seed> --out ... --three-phase true --teacher aq-public-strategy-selection --features-v2 true`。公開情報のみ、60合法候補・全4勝完走・teacher ID/versionを確認。初回で失敗した場合は評価や学習に進まず原因を修正
2. 旧v2学生と新しい模倣学生を、同一新教師の未使用データで評価。モデル/optimizerの初期条件を統制し、選出のTop1/Top10/normalized CEだけでなく教師の相対順位/utility regretを別計測する。新教師の模倣結果を旧教師の3.41%/4.55%と直接差分比較しない
3. **ドラフト**：現`chooseStrategyTeam`教師は複雑なチーム役割・相互作用を扱うが、studentの16D集約特徴は不足。まず合法候補全体の教師スコア・近接候補の効用差・候補数/難例を診断。無意味なTop1一致強制を避ける。次に新feature schemaで役割補完・相性・予算・メガ機会を表現し、既存checkpointは固定保全する
4. 改善を1要因ずつ分け、`旧教師+旧特徴`／`新教師+旧特徴`／`旧教師+新特徴`／`新教師+新特徴`／模倣→R-NaD／教師なしR-NaDを比較
5. 16実Match smoke→256実Match screening→新しい独立seed・両席での十分なheld-out。比較時にbattle Actorを同一凍結モデルにしてphase効果を確認。固定AQ-Base-001と旧モデルの両方へ直接対戦する

## 境界

- PR #6はDraft。既存v2/v3の学習済みモデル・checkpoint・旧教師の既定動作を変更しない
- 相手非公開セット・技・持ち物はActor/教師へ入力禁止。R-NaDは受理済みtrajectoryのみ学習
- GitHub-hosted Ubuntu CPU、重い研究Workflowはmanual-only。同一seriesにつき1 writer
- CI成功、受理済み教師Match成功、学習成功、実戦強度の改善、有意差、独立レビューは**別々の証拠**
