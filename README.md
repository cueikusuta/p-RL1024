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

## 実行基盤の性能検証

`node scripts/benchmark-engine.mjs` はRL-Deepモデルを9試合学習し（実際の重みを生成）、**変更前の推論・最適化後の逐次推論・最適化後の並列推論**を同じ64個の seed、完全4勝Matchで比較します。スコア・ラウンド履歴・アクション数の結果digestが全方式で一致しなければエラーとします。従来の初期モデル8試合のdigestは `test/parallel-frozen-deep.test.js` で別途保全します。

`node scripts/benchmark-training.mjs` は、同じseed・同じ24試合で、元のActor-Criticと最適化版の**逐次オンライン学習**を比較します。事前の同条件ウォームアップ後、試合履歴と最終学習モデルのSHA-256まで一致しなければ失敗とします。**学習途中の勾配更新を並列化していません**。

初回の学習済みモデルによる評価（GitHub-hosted Linux / Node 22.14.0、4ワーカー、64 Match）は、変更前逐次28.86秒、最適化後逐次4.93秒、最適化後並列3.47秒で、変更前比 **8.32倍** でした。実測の出典は [Actions run #43650315138](https://github.com/cueikusuta/p-RL1024/actions/runs/43650315138) です。これは特定モデル・環境における結果であり、他のモデル・マシンへの保証ではありません。

独立した凍結モデル評価のみ `scripts/batch-frozen-deep.mjs` の Worker Threads で並列化できます。従来の4 Match smoke は既定で逐次実行し、出力と検証契約を維持します。長い評価（正式な強さの証拠とは扱わない）を検証する例：

```sh
node scripts/evaluate-deep.mjs --model artifacts/rl-deep/training.json --purpose throughput --blocks 32 --workers 4 --out artifacts/rl-deep/throughput-32.json
```

worker 数と wall time は CPU 構成・負荷に依存するため、同一ホスト・同一 seed・同一 Node バージョンで比較してください。ベンチマークは強さ・汎化性能の証拠ではありません。
