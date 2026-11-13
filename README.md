# p-RL1024

Pokémon の対戦を題材にした、ファンメイドの機械学習実験です。作品に関する権利とデータの権利は、それぞれの権利者に帰属します。公式とは関係ありません。

現在は I975 の SARSA 学習と frozen smoke 評価を収録しています。これは評価経路の検証であり、モデルの強さを正式に評価するものではありません。画面、通信、サーバー、画像、音声は収録していません。

## Actions で実行する

**Actions → I975 frozen smoke and training → Run workflow** を選びます。`matches` に 9〜256、`seed` に 1〜4294967295 を指定できます。完了後、run の `i975-…` artifact に学習モデル、評価2回分、結果照合と8試合ごとの checkpoint が保存されます。artifact の保持期間は14日です。必要なモデルと評価結果は期限前にダウンロードしてください。

同じブランチの実験は同時に実行されません。待機中の run は別の run に置き換えられる場合があります。重複起動を避け、実行状態を確認してください。キャンセルやタイムアウトは成功ではありません。途中保存が存在しても、最後の試合まで完了したとは限りません。

## ローカルで実行する

Node.js 22.14.0 が必要です。追加の npm パッケージは不要です。

```sh
node --test test/pure-rl-linear.test.js test/pure-rl-phases.test.js test/pure-rl-acceptance.test.js test/verify-evaluation.test.js
node scripts/run-experiment.mjs
```

環境変数 `RL_MATCHES` と `RL_SEED` で試合数と seed を指定できます。出力先は `artifacts/i975/` です。比較結果にはモデルのファイル hash と、評価結果を正規化した digest を記録します。元コードが出力する outcome digest と、正規化 digest は別の値です。

`provenance/manifest.json` は元の private リポジトリのコミットと、取り込んだファイルの SHA-256 を記録します。実験結果を別環境と比較するときは、同じモデルの bytes、seed、引数、対戦相手を使ってください。Windows と Linux の一致はまだ確認されていません。
