# Issue #5｜R-NaDリーグの前向き・seed対応あり評価契約（2026-11-19）

目的：リーグで得られた世代・候補モデルの**小さいが再現可能な完全4勝Match改善**を測る。従来のseed/両席交換は維持し、今後の新規確認試験の統計解析を過度に保守的な範囲限定Hoeffdingから、実測した**独立seedブロック間分散を使う対応ありStudent-t**へ切替える。**学習アルゴリズム/optimizer/モデル/既存対戦ログを変更しない。**

## 方針・レベルの固定

- 先に確定済みの [FUTURE_EVALUATION_CONFIDENCE_POLICY_2026-11-17.md](FUTURE_EVALUATION_CONFIDENCE_POLICY_2026-11-17.md) に従い、**新規実験のfamilywise信頼水準は原則80%（α=0.20）**。個々の区間を無補正で同時にChampion判断しない。95%を要する新規実験も明示可能だが、結果を見て選択してはならない。**80%信頼水準は検出力80%という意味ではなく、比較family全体の偽陽性許容が最大20%である**。
- **512独立seedブロック（両席で1,024完走Match）以上を正式な最終確認の最低条件**とする。ただし512ブロックで+1%が必ず検出できるわけではない。開発pilotのブロック間SDと検出目標から必要数を事前決定し、必要ならより多く実施する。128～511ブロックの結果はdevelopment診断のみ。
- seedブロックを独立標本とし、**同seed両席の2対戦は独立2標本と数えない**。戦闘ラウンド/行動/学習バッチ/教師Top1/学習リーグの内部勝率も、正式な4勝先取Match勝率の標本数に使わない。
- **直前世代/現Championとの直接H2H**：各seedで両席を交換し、新世代の2試合勝数を2で割った `s_i∈{0,.5,1}` を観測。主要帰無 `E[s_i]=.5`。この結果は「相手の旧世代に直接勝てるか」の判定に限る。
- **共通凍結相手への独立比較**（例G11とG10をそれぞれAQと対戦）：各seed/両席の勝率 `s_{new,i}`・`s_{old,i}` を同一seedで対応させて `d_i=s_{new,i}-s_{old,i}∈[-1,1]`、主要帰無 `E[d_i]=0`。**新旧の対AQ勝率の差は直接H2H勝率ではない**。
- 主要検定と同時区間は、ブロック平均のStudent-t（`n-1`自由度）、**事前比較件数mに対しBonferroniの各両側α/m**。Holm等は別plan versionで明示導入。正規近似や片側区間への事後的な切替禁止。
- 追加の探索的頑健性検査としてseedブロック単位の符号検定（0差ブロックは除く）を記録する。Student-tの区間は**分布自由な有限標本保証ではない**。分散が0、分布が極端に偏るなどの条件は回帰試験・別確認で検証する。探索的検定結果で採否をすり替えない。
- **多世代/複数リーグ相手**を同じcohortで評価する場合、計画時に全比較を列挙しfamilywise補正。確認コホートで最高の点推定モデルを見て後選択しない。事前に候補を1つへ固定するか、当初から候補すべてを補正対象とする。
- CI下限が0（H2Hは50%）を上回っても自動Champion昇格しない。追加の実用的改善幅（0ptでも可）、AQ強度、旧世代への有害な退化の有無、20秒・合法/秘匿/交代ループ/推論メモリ・出典SHA・完全4勝Match・別ChatGPT Chat独立レビューを別に検証する。

## 実装

- 計算：[`scripts/rnad/paired-league-stats.mjs`](../../../scripts/rnad/paired-league-stats.mjs)
- **次の新規評価用**fail-closed集計：[`scripts/rnad/paired-league-eval.mjs`](../../../scripts/rnad/paired-league-eval.mjs)
- 契約・退行検証：[`test/rnad-paired-league-eval.test.js`](../../../test/rnad-paired-league-eval.test.js)
- ランナー：GitHub-hosted **ubuntu-latest**。学習用セルフホストへの移行なし。既存S0～S2の学習・今までの固定launch workflowは不変。
- 新しく計画するたびに`rnad-i5-prospective-paired-league-plan-v1` JSONを**評価開始前にIssue #5へ登録しSHA固定**する。最低限`stage`（development/final）、`evaluation_type`（shared-opponent/direct-head-to-head）、比較するモデルID、評価source/engine SHA、benchmark/Champion ID、モデル重みSHA、完全未使用のseed_start/seed_blocks、学習・development・消費済holdoutの`excluded_seed_ranges`、familywise α、比較family、minimum_practical_gain、固定試合数と中間閲覧/停止規則、想定の効果量と検出力を記す。
- 原入力は各候補・各モデル版の `rnad-i5-frozen-paired-match-evidence-v1`：2席×各seedの`completed:true`、4勝Matchの`score`、`rounds`、`accepted_decisions`、`won`、`source_sha`/ `engine_sha`/ `model_sha256`、評価中no-trainingを厳密照合する。複数shardを結合可能だが欠落/重複/体裁不整合/同一版の重み変化は拒否。
- 集計CLI：`node scripts/rnad/paired-league-eval.mjs PLAN.json REPORT.json SHARD1.json [SHARD2.json ...]`。出力が `statistical_gate_passed:true` でも**Champion書込・学習再開・PRマージの操作は一切行わない**。原Artifactが実GitHub Runで受理されたMatchかどうかは、別のsource-bound収集/Artifact SHA検証で確証する必要があり、自己申告JSONだけで本番昇格してはならない。
- 完了した学習世代のチェックポイントを**評価時に凍結**し、同一モデル版を両席・全shardで使う。開発評価で使ったseedを最終の未使用確認へ再利用しない。学習中のリーグ戦は非定常・方策変更中なので正式な独立確認データに転用しない。

## 既存の過去研究は変更しない

- `docs/research/rnad/G012_AQ_LAUNCH_2026-11-19.json` の過去固定実験（24,576完全Match、95%・6比較）、過去20モデル研究の190比較/95%ゲート、旧`pairedHoeffding95`・旧`aggregate-eval.mjs`の出力schema/実験結果は変更・再解釈しない。
- 既存の[独立レビューP1](https://github.com/cueikusuta/p-RL1024/pull/6#pullrequestreview-5938960571)：自己申告だけでリーグ開始/昇格できる不正ゲートの修正と、信頼済み実証拠との照合は依然必須。この評価器を迂回口にしない。
- 現在の #5 / Draft PR #6 で開発を継続。実装のテストが成功しても、**新方式で未使用の大規模最終対戦を実際に完了したことや、独立レビュアーが承認したことにはならない**。別の方策が勝ったという断定も行わない。

## 新規plan例（書式のみ／未実行）

```json
{
  "schema": "rnad-i5-prospective-paired-league-plan-v1",
  "stage": "development",
  "evaluation_type": "shared-opponent",
  "source_sha": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "engine_sha": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  "opponent_id": "AQ-Base-001",
  "seed_start": 7100001,
  "seed_blocks": 512,
  "familywise_alpha": 0.2,
  "confidence_level": 0.8,
  "method": "paired-seed-block-student-t-bonferroni",
  "fixed_sample_size": true,
  "model_selection_before_holdout": true,
  "automatic_champion_promotion": false,
  "minimum_practical_gain": 0,
  "excluded_seed_ranges": [{"start":6200001,"count":128}],
  "comparisons": [{"candidate_model_id":"G11","baseline_model_id":"G10"}]
}
```

この例のSHAとseedはプレースホルダーであり、**利用可能な実コホートの証明ではない**。出自・モデル・未使用seedリストを別途実証してから新runを起動する。
