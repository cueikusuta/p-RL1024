# p-RL1024

Pokémon の対戦を題材にした、ファンメイドの機械学習実験です。作品に関する権利とデータの権利は、それぞれの権利者に帰属します。公式とは関係ありません。

RL-Linear（I975、SARSA）と RL-Deep（I976、Battle Action のみの小型 MLP Actor-Critic）を収録しています。どちらも smoke 評価であり、モデルの強さを正式に評価するものではありません。画面、通信、サーバー、画像、音声は収録していません。

## Actions で実行する

**Actions → RL model smoke and training → Run workflow** を選びます。`model` と `opponent` の組合せに **RL-Linear / V2E** または **RL-Deep / RL-Deep** を指定します。`matches` は 9〜256、`seed` は 1〜4294967295 です。RL-Deep は公開観測と自軍情報から合法な Battle Action を選び、Basic のドラフト・選出と凍結自己対戦で学習します。AQ-Base の重みは含まれておらず、AQ-Base 対戦は実装していません。

完了後、run の `i975-…` artifact に学習モデル、評価2回分、結果照合と8試合ごとの checkpoint が保存されます。artifact の保持期間は14日です。必要なモデルと評価結果は期限前にダウンロードしてください。

同じブランチの実験は同時に実行されません。待機中の run は別の run に置き換えられる場合があります。重複起動を避け、実行状態を確認してください。キャンセルやタイムアウトは成功ではありません。途中保存が存在しても、最後の試合まで完了したとは限りません。

## ローカルで実行する

Node.js 22.14.0 が必要です。追加の npm パッケージは不要です。

```sh
node --test test/pure-rl-linear.test.js test/pure-rl-phases.test.js test/pure-rl-acceptance.test.js test/verify-evaluation.test.js
node scripts/run-experiment.mjs
node scripts/run-experiment.mjs --model RL-Deep --opponent RL-Deep
```

環境変数 `RL_MATCHES` と `RL_SEED` で試合数と seed を指定できます。出力先は `artifacts/i975/` または `artifacts/rl-deep/` です。比較結果にはモデルのファイル hash と、評価結果を正規化した digest を記録します。元コードが出力する outcome digest と、正規化 digest は別の値です。

中断した実験をローカルで再開するには、artifact 内の checkpoint を取得し、元と同じ snapshot、予定試合数、seed、学習設定を指定します。次は予定9試合、seed 975020、8試合目からの例です。

```sh
node scripts/dev/simulator/pure-rl-sarsa-train.mjs --all-phases true --matches 9 --seed 975020 --checkpoint-every 8 --resume artifacts/i975/training-m8.json --out artifacts/i975/resumed.json
```

RL-Deep の同条件の再開には `scripts/train-deep.mjs` を使います。

```sh
node scripts/train-deep.mjs --matches 9 --seed 975020 --checkpoint-every 8 --resume artifacts/rl-deep/training-m8.json --out artifacts/rl-deep/resumed.json
```

技術系統と特徴量の対応は `models/registry.json` にあります。RL-Deep は Node.js の新規実装で、v2-E 教師や #877 の学習コードを使用していません。正式な強さの比較やドラフト・選出の深層RL化は含みません。

`provenance/manifest.json` は元の private リポジトリのコミットと、取り込んだファイルの SHA-256 を記録します。実験結果を別環境と比較するときは、同じモデルの bytes、seed、引数、対戦相手を使ってください。2026-11-14 に元コミットの Windows 実行と公開 snapshot の Windows・GitHub-hosted Linux 実行で、9試合学習後のモデルと frozen smoke の結果 digest が一致しました。これは smoke 条件の検証です。
