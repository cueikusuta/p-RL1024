# p-RL1024

Pokémon の対戦を題材にした、ファンメイドの機械学習実験です。作品に関する権利とデータの権利は、それぞれの権利者に帰属します。公式とは関係ありません。

Pokémon Showdown 参照 commit のデータや、参照して実装したコードに関する MIT の著作権表示とライセンス本文は [LICENSES/pokemon-showdown-MIT.txt](LICENSES/pokemon-showdown-MIT.txt) を参照してください。第三者のデータや名称に関する権利は別途あります。出典と未確認事項は [provenance の調査記録](provenance/catalog-usage-audit-2026-11-14.md) を参照してください。

RL-Linear（I975、SARSA）と RL-Deep（I976、Battle Action のみの小型 MLP Actor-Critic）を収録しています。どちらも smoke 評価であり、モデルの強さを正式に評価するものではありません。画面、通信、サーバー、画像、音声は収録していません。

## Actions で実行する

**Actions → RL model smoke and training → Run workflow** を選びます。`model` と `opponent` の組合せは **RL-Linear / V2E**、**RL-Deep / RL-Deep**、または **RL-Linear / AQ-Base-001**、**RL-Deep / AQ-Base-001** を指定します。`matches` は 9〜256、`seed` は 1〜4294967295 です。AQ-Base-001 指定時は学習と評価の両方で同じ凍結 AQ-Base を相手とします。RL-Deep は公開観測と自軍情報から合法な Battle Action を選び、ドラフト・選出は Basic です。

完了後、run の `i975-…` artifact に学習モデル、評価2回分、結果照合と8試合ごとの checkpoint が保存されます。artifact の保持期間は14日です。必要なモデルと評価結果は期限前にダウンロードしてください。

同じブランチの実験は同時に実行されません。待機中の run は別の run に置き換えられる場合があります。重複起動を避け、実行状態を確認してください。キャンセルやタイムアウトは成功ではありません。途中保存が存在しても、最後の試合まで完了したとは限りません。

## ローカルで実行する

Node.js 22.14.0 が必要です。追加の npm パッケージは不要です。

```sh
node --test test/pure-rl-linear.test.js test/pure-rl-phases.test.js test/pure-rl-acceptance.test.js test/verify-evaluation.test.js
node scripts/run-experiment.mjs
node scripts/run-experiment.mjs --model RL-Deep --opponent RL-Deep
node scripts/run-experiment.mjs --model RL-Deep --opponent AQ-Base-001
```

環境変数 `RL_MATCHES` と `RL_SEED` で試合数と seed を指定できます。出力先は `artifacts/i975/`、`artifacts/rl-deep/` または `artifacts/<モデル>-vs-aq/` です。比較結果にはモデルのファイル hash と、評価結果を正規化した digest を記録します。元コードが出力する outcome digest と、正規化 digest は別の値です。

中断した実験をローカルで再開するには、artifact 内の checkpoint を取得し、元と同じ snapshot、予定試合数、seed、学習設定を指定します。次は予定9試合、seed 975020、8試合目からの例です。

```sh
node scripts/dev/simulator/pure-rl-sarsa-train.mjs --all-phases true --matches 9 --seed 975020 --checkpoint-every 8 --resume artifacts/i975/training-m8.json --out artifacts/i975/resumed.json
```

RL-Deep の同条件の再開には `scripts/train-deep.mjs` を使います。

```sh
node scripts/train-deep.mjs --matches 9 --seed 975020 --checkpoint-every 8 --resume artifacts/rl-deep/training-m8.json --out artifacts/rl-deep/resumed.json
```

AQ-Base 対戦を再開する場合は、元と同じ `--model`・試合数・seed と snapshot で `scripts/train-vs-aq.mjs` の `--resume` に checkpoint を指定します。

技術系統と特徴量の対応は `models/registry.json` にあります。RL-Deep は Node.js の新規実装で、v2-E 教師や #877 の学習コードを使用していません。正式な強さの比較やドラフト・選出の深層RL化は含みません。

AQ-Base-001 の凍結重みは本体台帳の SHA-256 と照合しています。AQ 対戦には別の固定エンジンスナップショット `aq-runtime/` を使います。これは従来の I975 smoke に用いる snapshot とは別のため、異なる条件の結果を直接比較しないでください。

`provenance/manifest.json` は元の private リポジトリのコミットと、取り込んだファイルの SHA-256 を記録します。実験結果を別環境と比較するときは、同じモデルの bytes、seed、引数、対戦相手を使ってください。2026-11-14 に元コミットの Windows 実行と公開 snapshot の Windows・GitHub-hosted Linux 実行で、9試合学習後のモデルと frozen smoke の結果 digest が一致しました。これは smoke 条件の検証です。

## フェーズごとに異なるAIを使う（opt-in）

元の I975/I976 の学習・評価経路と `provenance/manifest.json` によるスナップショット固定は変更しません。新しい `scripts/dev/simulator/phase-match.js` の `runPhaseMatch` を呼んだ場合だけ、両座席のドラフト・選出・戦闘AIを独立に指定できます。

```js
import {runPhaseMatch} from './scripts/dev/simulator/phase-match.js';
import {createBuiltinPhaseAgents,sessionPhaseAgent}
  from './scripts/dev/simulator/phase-agents.js';
import {createModel,createSession} from './models/deep.mjs';

const deep=createSession(createModel(975020),{train:false});
const result=runPhaseMatch({seed:975020,agents:[
  createBuiltinPhaseAgents({
    draft:'strategy',selection:'tactical',
    battle:sessionPhaseAgent('rl-deep',deep)
  }),
  createBuiltinPhaseAgents({
    draft:'basic',selection:'strategy',battle:'intermediate'
  }),
]});
console.log(result.winner,result.score,result.agentIds);
```

上記の `createModel` は初期値モデルのデモであり、学習済みRL-Deepの性能を示すものではありません。利用可能な内蔵アダプタはドラフト `basic / strategy`、選出 `basic / strategy / tactical`、戦闘 `basic / intermediate` です。ほかのAI（RL-Linear、RL-Deep、V2E、AQ-Baseなど）は適切な同期セッションや独自ハンドラを指定して接続します。すべてが内蔵選択肢に登録済みという意味ではありません。

各フェーズのハンドラは `{id, select({view,context,rng,seat,round}), accepted?, finish?}` を実装します。`select` は既存のアクションを直接返すか、`{action,handoff}` を返します。後者の `handoff` は最大4096文字のJSONオブジェクトで、ドラフト／選出／戦闘にそれぞれ独立して記録されます。直前のフェーズの `context.current`、過去ラウンドの `context.completed`、モデルが残した `context.handoff` を別のAIから参照できます。`context.schema` は `p-rl1024-phase-pipeline-v1` です。

共通コンテキストは座席ごとに分離され、権威Matchが受理した自軍のドラフト・選出・戦闘アクションのみ記録します。自軍の確認済みセットと相手の公開種族・観測可能なHP割合等は共有しますが、相手の未公開セット、未公開技、真の内部状態は扱いません。AIが返した不正アクションはMatch/Battle側で拒否され、`accepted` や学習の `commit` は呼ばれません。非同期AIは現状未対応です。

検証：

```sh
node --test test/phase-pipeline-contract.test.js test/phase-pipeline-integration.test.js
```

## 実行基盤の性能検証

`node scripts/benchmark-engine.mjs` はRL-Deepモデルを9試合学習し（実際の重みを生成）、**変更前の推論・最適化後の逐次推論・最適化後の並列推論**を同じ64個の seed、完全4勝Matchで比較します。スコア・ラウンド履歴・アクション数の結果digestが全方式・各反復で一致しなければエラーとします。**3回の反復実測で旧逐次に対する最適化並列の速度比中央値が2.00未満ならCI失敗**にします。従来の初期モデル8試合のdigestは `test/parallel-frozen-deep.test.js` で別途保全します。

`node scripts/benchmark-training.mjs` は、同じseed・同じ24試合で、元のActor-Criticと最適化版の**逐次オンライン学習**を比較します。事前の同条件ウォームアップ後、**3回反復して速度比の中央値が2.00未満ならCI失敗**にします。各回の試合履歴と最終学習モデルのSHA-256は完全一致が必要です。**学習途中の勾配更新を並列化していません**。

初回の学習済みモデルによる評価（GitHub-hosted Linux / Node 22.14.0、4ワーカー、64 Match）は、変更前逐次28.86秒、最適化後逐次4.93秒、最適化後並列3.47秒で、変更前比 **8.32倍** でした。実測の出典は [Actions run #43650315138](https://github.com/cueikusuta/p-RL1024/actions/runs/43650315138) です。これは特定モデル・環境における結果であり、他のモデル・マシンへの保証ではありません。

独立した凍結モデル評価のみ `scripts/batch-frozen-deep.mjs` の Worker Threads で並列化できます。従来の4 Match smoke は既定で逐次実行し、出力と検証契約を維持します。長い評価（正式な強さの証拠とは扱わない）を検証する例：

```sh
node scripts/evaluate-deep.mjs --model artifacts/rl-deep/training.json --purpose throughput --blocks 32 --workers 4 --out artifacts/rl-deep/throughput-32.json
```

worker 数は利用可能な論理CPU数以下かつ16以下に制限し、巨大モデルのworkerコピー用メモリの保守的上限も検証します。wall time は CPU 構成・負荷に依存するため、同一ホスト・同一 seed・同一 Node バージョンで比較してください。ベンチマークは強さ・汎化性能の証拠ではありません。

## RNAD / 機械学習：実験開始前の共通検証条件

AI/学習セッションは、大規模学習、教師ありのモデル変更、R-NaD更新、強度評価を始める前に、[pokemon-battle-factory の AI/ML実験・強度判定共通契約 v1](https://github.com/cueikusuta/pokemon-battle-factory/blob/docs/1084-ml-validation-contract/docs/guides/ai-ml-validation-gates.md)を確認し、**仮説・独立比較・seed分離・完走/交代ループ・教師/推論モード・最終判定ゲート**を Issue #5 / PR #6 に事前登録してください。この文書は本体側PRでの規約案なので、develop統合まではそのPRのHEAD/独立レビュー状況を照合してください。既存RNADモデルの学習成果を、文書追加だけで強化済みとみなしてはいけません。

RNAD固有の過去負例と入力特徴量の現状は、[FAILURE_FEATURE_GAP_AUDIT_2026-11-16.md](docs/research/rnad/FAILURE_FEATURE_GAP_AUDIT_2026-11-16.md)および[原Runの横断監査](docs/research/rnad/EVIDENCE_AUDIT_2026-11-16.md)を参照。両リポジトリでコード・重み・Runner設定を共有せず、検証カテゴリと証拠契約のみ共有します。

## RNAD-001 (Issue #5): GitHub-hosted runner policy

RNAD-001 の開発・小規模検証・本学習・評価は **GitHub Actions の GitHub-hosted runner（`ubuntu-latest`）のみ**を使用します。**セルフホスト runner は使用しません**。これは2026-11-15のユーザー指定による最新の運用方針で、旧計画のセルフホスト推奨より優先されます。

研究の設計と未完了ゲートは [docs/research/rnad/PLAN.md](docs/research/rnad/PLAN.md) を参照してください。学習は明示的な `workflow_dispatch`、試合数・時間・同時書込制限、コードSHAとCheckpoint整合性確認、独立レビューの品質ゲートを通してのみ実行します。短期GitHub Actions artifactを永続Checkpointの正本として扱わず、学習成果と評価結果を区別します。
