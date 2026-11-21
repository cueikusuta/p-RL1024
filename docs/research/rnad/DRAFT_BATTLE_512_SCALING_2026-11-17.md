# I5｜ドラフト・戦闘模倣の追加学習効果：256→512完全教師Match研究

## 目的
ユーザーの指示「まずは学習を進めればドラフト・戦闘の教師Top1模倣率が上げられるか確認する」ための、既存Issue #5／Draft PR #6に属する一回限りの固定研究。**目的は追加学習の効果を検証することであり、実戦勝率やリーグ開始条件達成を証明することではない。**

従来別個の研究モデルでは、ドラフトDeepSets soft256-e6の教師Top1 **41.60%**、戦闘v5 battle-focus256のTop1 **75.84%**（ともに既用128完全教師Match）が報告されている。旧128→256のドラフトTop1は31.64→41.60%だったが、旧128教師Matchは候補選択に再利用されており、現在の研究の統計確証標本ではない。

## 研究対象と事前固定条件
- 権威ゲーム・教師・feature source を過去の成功データの**exact Git commit `022aa9cd5b4ac37514df1d845ea00c11c3c13a5c`**で凍結。教師 `rnad-aq-public-strategy-selection-v1`、特性v5、合法候補、utility labels、draft review、完全4勝Matchの全決定を検証。教師/ゲーム/特徴量を変えない。
- 過去の学習seed `2471001..2471256`（256完全Match）は[Run #44032488678](https://github.com/cueikusuta/p-RL1024/actions/runs/44032488678)の成功済みArtifactをSHA/provenance照合して再利用する。過去に使用済みの開発seed `2472001..2472128` は**今回の教師学習にも最終評価にも再使用しない**。
- 新規学習専用 **256完全教師Match seed 2520001..2520256**。旧256+新規256でドラフトを同じDeepSets構造・soft utility順位蒸留・6epochs（従来と同一学習関数・固定初期乱数）で**ゼロから512完全Match学習**し、凍結旧soft256-e6と比較。追加したハイパーパラメータ探索はない。
- 戦闘は凍結[recovery Run #44037414890](https://github.com/cueikusuta/p-RL1024/actions/runs/44037414890)の既存v5 battle-focus256 6epochs学習済み重みをSHA固定して初期化。**新規256完全教師Matchにさらに6epochs**、phase-balanced `battle-focus`（1:1:4）、`--utility-distillation --utility-mix 0`で従来と同一hard CE。**オプティマイザー/RNGの完全checkpointではなく、重みだけを継承して新規Adam最適化**。旧256と新256の合計512完全教師Match由来の重みを得るが、厳密な「一気に512Match学習」には相当しない。この差を結果に明記する。
- **新規・モデル選択未使用の512完全4勝教師Match seed 2522001..2522512**で、旧256モデルと新モデルを**全く同じ教師受理済み局面**で比較し、全モデルをこのholdoutの成績を見る前に固定。
- 指標：フェーズ別教師Top1（判断数加重）、独立complete-Match seedごとのTop1平均、対応ありseed-cluster差の10,000 bootstrap近似**両側95%CI**。改善の支持はCI下限が **0超**、悪化の支持は上限が0未満、CIが0を含むなら**結論保留**。少数試合/判断を独立扱いしない。参考として3相Bonferroni補正の80%片側Hoeffding下限とDraft >50%／Battle ≥80%を報告するが、**同一統合3相policyのリーグゲートとは別物**。
- スケーリング研究の初回失敗時は同じRunで復旧を優先し、新しい成果がないうちにseedやモデルを見直して再選択しない。過去Runの教師Matchは再生成しない。研究専用GitHub-hosted `ubuntu-latest` CPU、1writer・fail-fast・先行Test通過後のみ重い処理。
- 成果は `scripts/rnad/draft_battle_learning_512.py`・`test/rnad_draft_battle_learning_512_test.py`・専用Workflow `rnad-i5-draft-battle-512-scaling.yml` のArtifactとして重みSHAと全seedを保全。独立レビューで指摘済みの`league_entry_gate.py`のP1（自己申告評価証拠の信頼）を修正/再レビューするまでは**研究結果をリーグ起動入力にしない**。

## 合格・境界
1. 旧256対新512の比較は**ドラフト・戦闘の各フェーズごと**に同じ512個の新規完全Matchを使うため対応がある。効果判定は両側95%CIに基づく。
2. 教師模倣Top1の改善は、元AQ-Base-001への実戦勝率改善の確証ではない。ドラフト・選出・戦闘の別学生を集めた結果も同一統合政策のリーグゲート合格ではない。
3. 現状のリーグ開始条件 Draft >50%、Selection ≥50%、Battle ≥80% は、**同一統合政策・同一未使用権威Match・三相補正下限**すべてで満たす必要がある。Champion、リーグ学習、PRマージは禁止。
