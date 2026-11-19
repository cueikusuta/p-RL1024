# Issue #5 — 次期共有リーグ学習・対応あり評価の実行準備（2026-11-19）

対象：[Issue #5](https://github.com/cueikusuta/p-RL1024/issues/5) / [基盤PR #6](https://github.com/cueikusuta/p-RL1024/pull/6) / [高速化PR #8](https://github.com/cueikusuta/p-RL1024/pull/8)。**これは次回実行の準備記録であり、本格学習の開始、Champion昇格、性能優位の証明ではない。**

## 維持する学習契約

- R-NaD / NeuRD / V-trace、ドラフト・選出・戦闘の統合Policy、権威Matchが受理した行動だけを使用する。
- 1世代＝20 Actor × 32独立seed × 両席入替＝**1,280完全4勝Match**。32完全Match/更新×40更新、Learnerは1つの共有Optimizer。
- Actorの自己対戦／過去世代／固定強敵の比率はmanifestを正とし、学習前に固定する。Championモデルと旧世代の重みは更新しない。
- 全20 Actorの完全性・seed重複・席・合法性・出自を検証した後だけLearner更新。障害時は失敗Actorだけを元のseedで再実行し、strict 20/20を回復できなければ停止する。
- checkpointのMain/Target/Magnet/Optimizer/Scheduler/RNG、source SHA、engine SHA、モデルSHA、世代・受理Match IDを保存・照合。既存学習済み重みは上書きしない。
- 研究系Issue #5のランナー決定に従い、学習・評価は**GitHub-hosted ubuntu-latest CPU**のみ。self-hosted/GPU/larger runnerに自動切替しない。同時Actorは最大20、他の実験の競合とArtifact転送を抑制する。

## 高速化：現時点で実証した範囲

- [Run #44447994233](https://github.com/cueikusuta/p-RL1024/actions/runs/44447994233) は、**古い固定G0の実受理1,280 Matchと初期状態**を使って、同じ40回の学習更新を歴史的リプレイとして比較した研究実行。source 5554952d2d9d9136b163adc49ee48400dda721f3 上で完走。
- reference scalar 40更新 **367.177秒**、CPU parallel-vector 40更新 **130.083秒**、比 **2.8226倍（単一試行）**、最終モデルの最大パラメータ絶対差 **6.720423698425293e-06**、bitwise同一ではない。研究ベンチマークであり、元の学習鎖へ新checkpointを接続したことにはならない。
- Actor収集・依存導入・20ジョブの待機・転送・復旧・checkpointアップロードを含む**世代総壁時計**の2倍改善は未確認。次回は `Actor開始→20/20検証→Learner 40更新→handoff完了` の各経過時間と総時間、runner-minutes、エラー率、試合/秒、モデル差分を記録する。
- PR #8 の高速経路（vector/parallel、selection埋込再利用、Actor永続プロセスなど）は**opt-in**で使い、strict parity、実受理Match、数値許容差、最新HEAD CIを満たした経路だけ採用する。ONNX Actorやtorch.compileの未検証経路を既定化しない。

## 追加のシミュレーション高速化（PR #8、品質ゲート未完了）

- **常駐Actor（①）**：1 Actorにつき単一Nodeを維持し、最大64完全MatchでSHA検証済みモデルを再利用。各Match・seatのGRU状態は新規生成し、試合別gzipは独立排他的に保存。単独計測64 Matchで約1.012倍に留まったため、倍率の保証はしない。
- **GRU推論（②）と割当削減（④）**：`infer.mjs` でGRUゲートscratch、および候補用のstate/action連結scratchとhidden scratchを同期呼び出し内だけ再利用。積和の加算順序は旧実装を維持し、両seat・`advance=false`・resetを含む独立スカラー参照との**厳密一致テスト**を追加。推論結果からscratchへの参照を返さない。
- **合法行動生成と軌跡（③）**：`validateRnadLegalActions` の厳密な重複検査で作成したJSON署名を内部collectorの包含判定にも再利用。外部から受け取る合法actionのclone・公開観測のallowlist・authority receipt検証を削除しない。観測projectionとphase pipelineの権限境界は従来どおり。
- **Deep Setsドラフト推論の追加改善（②・④）**：8実受理MatchのCPUプロファイルで`scoreEncoded`の自己サンプル69.84%が最大ホットスポット。`draft-deepsets-infer.mjs`の候補ごとのrho入力・中間層配列を同一`scoreEncoded`呼び出し内で再利用し、6体の重複判定でSetを生成しない。非ゼロ重み・多数合法候補で旧スカラー実装との厳密一致テストを追加し、最新HEADで検証する。これによりスコア・合法候補・試合出力を変更してはならない。実測速度は別途CIで測定する。
- **履歴I/O（⑤）**：既存PR #8の厳密gzip・SHA検証と一時plain JSON／tensor cacheを維持。未検証cacheを既定にしない。
- **Actor内並列（⑥）**：`shared_league_collect.py --match-workers 1|2|4` を追加。既定1、2/4は任意のCPU余力があるときのみ明示指定。各プロセスは互いに重複しないseed/seat集合を常駐処理し、失敗時にはすべての完了Artifactを再検証して不足分のみ再試行。20 Actorの**20/20 barrier、共有Learner単一Optimizer、1,280 Match/40更新は不変**。同時に20 Actor×4を無条件で有効化してはならない。
- **検証**：CIは単一Node×64、2Node×64、各Matchごと単独Node×64について全ての権威受理軌跡を一致照合し、各ルートの所要時間を記録する。これらは速度の診断であり、有意な実end-to-end改善を単一測定から主張しない。新コミットのソースSHAに合うbootstrapでのみ実学習する。

## 次の学習runに進む前の必須チェック

- [ ] PR #8 の最新HEADテストと実Matchの受理一致を確認し、**別ChatGPT Chatセッション**による独立レビューで指摘を解消する。PR #6側のP1・既存フェーズentryゲートも未解決を成功扱いしない。
- [ ] **新コードSHAに対応したbootstrap**を作る。過去のsource-pinned G0/G1/G2 checkpointを新コードで無断継続せず、同一性が監査された移行または新bootstrapを用いる。古いArtifactの存在・SHA・期限を再確認する。
- [ ] 1世代20 Actor・共有Learnerの**実end-to-end壁時計**を旧方式と同条件で測る。単一ベンチマーク比を全体速度と呼ばない。
- [ ] 合法性・秘密情報・Actor/shard完全性・新旧seed不重複・Optimizer継続・GPU無しを再確認する。
- [ ] 研究runと独立評価runのseed領域、出力場所、Artifact保持/長期保存先、source/engine/model SHA、予算と失敗時停止条件を**起動前**に固定する。
- [ ] 合格したときだけ次の複数世代を実行。性能が上がらない世代の無制限延長はしない。試合数・更新数だけでは採用判定しない。

## 対応あり評価：世代モデルを凍結し、完全Matchで比較

既存の [対応あり評価契約（親PR #6）](https://github.com/cueikusuta/p-RL1024/blob/feat/5-rnad-stage0-contracts/docs/research/rnad/PAIRED_LEAGUE_EVALUATION_2026-11-19.md)、`paired-league-run.mjs` / `paired-league-eval.mjs` / `paired-league-stats.mjs` を再利用する。旧評価結果は再解釈しない。

1. **評価計画を事前登録**。candidate、旧世代/Champion、AQ-Base、source SHA、engine SHA、モデルmanifest SHA256、未使用seed cohort、除外seed、固定標本数、比較family、多重比較補正、実用的改善幅、停止規則をIssue #5に保存。実ファイルSHAとArtifactを確認するまで数字を捏造しない。
2. **直接H2H**：同seedで新世代対旧世代を両席入替。seedブロック `s_i=(2戦の新世代勝数)/2` と帰無値0.5で対応あり検定。
3. **AQ共通対照**：同seedで新旧をそれぞれ凍結AQと2席対戦。`d_i=s_new,i-s_old,i` と帰無値0で対応あり検定。これを直接H2H勝率として扱わない。
4. 解析標本単位は**独立seedブロック**。2席で試合数が倍になっても独立標本数は倍にしない。計画済み全比較をBonferroni補正。主要解析は両側paired Student-t、符号検定は感度検査。従来の計画方針でfamilywise 80%（alpha=.20）を基本とし、より厳格な95%を使う場合は結果を見る前に明記する。
5. **最終確認は最低512 seedブロック＝1,024完全Match／モデル対戦相手**。この下限だけで小差を判断せず、pilotのseed間標準偏差・検出したい差・比較数から必要ブロック数（必要なら2,048/4,096以上）を事前に計算する。128–511ブロックは開発診断のみ。探索済みseedは最終未使用holdoutに再利用しない。
6. 全seed・両席・4勝完走・受理行動数・weight SHA・engine/source SHA・評価中無学習・Artifact出自を検証し、欠損・重複・勝者不整合はfail-closed。
7. 数値上の統計ゲート達成でも自動昇格しない。対AQの水準、前Champion回帰、実戦20秒・合法性・秘匿情報・交代ループ、最新HEAD独立レビューを別ゲートにする。

## 実行順

**A.** PR #8最適化の修正・検証 → **B.** 新sourceでの完全状態bootstrap → **C.** 1,280 Match×1世代の実end-to-end性能測定 → **D.** 問題なければ複数世代継続・各世代を凍結保管 → **E.** モデルを事前選定して対応あり開発評価 → **F.** 未使用holdoutを事前登録して必要十分なブロック数の正式評価 → **G.** 独立レビュー/昇格判断。

この文書は**学習と統計検証の混同・小標本判断・旧source SHAの不正引継ぎを防ぐための準備契約**。実際のstartを示すRun ID、確定モデルSHA、評価seed範囲・CI成功は実行後のみ追記する。
