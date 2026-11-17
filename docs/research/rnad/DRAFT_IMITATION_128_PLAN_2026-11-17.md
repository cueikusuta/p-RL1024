# I5：ドラフト模倣率の改善（教師128Match、未使用64Match）

実験正本：Issue [#5](https://github.com/cueikusuta/p-RL1024/issues/5)、Draft PR [#6](https://github.com/cueikusuta/p-RL1024/pull/6)。この文書は**新規実験の仕様**。未完了Runを「改善済み」と誤報しない。

## 目的・現状

旧Deep Sets 32教師Match版は未使用8教師Match（80回のドラフト判断）で **Top1=16.25%、Top10=63.75%**。この80回の判断は独立80Matchではなく、十分な強度検証ではない。さらに旧16教師Match版を3,072seed×両席で固定N実戦確認したところ、v3に対して**−3.79pt**（95%同時CI [−7.26pt,−0.32pt]）と有意に劣った。**教師模倣の指標と実戦勝率は別物**。

今回は模倣自体の一般化性能を改善する。対象は **同じv1アーキテクチャ** を使った、新しい独立モデル。旧モデル/チェックポイント、既存の勝率統計を上書きしない。

## 実験1：凍結32Matchモデル vs 新128Matchモデル

- 凍結対照：過去[Run #43986137367](https://github.com/cueikusuta/p-RL1024/actions/runs/43986137367)の `deepsets-soft-full.json`、SHA256 `da186af7581485382b4f558f356ac285bbf071817239cba7271cb05615e3b037`。単なる再学習ではなく**実際の旧重みを復元**して同じ未使用局面でテストする。
- 新しい教師学習：受理された公開AQドラフト教師の**128完全Match**、seed `2461001..2461128`、4 epoch。旧v4可視情報・合法候補・単一座席の受理済み教師選択だけを利用。
- 新しい未使用ホールドアウト：**64完全Match**、seed `2462001..2462064`。候補の選別・学習・epoch決定に使わず、3モデルが同じ全局面を評価。1 Match中の複数ドラフト判断を別の独立標本として水増ししない。
- 3アーム：旧32教師Match soft2epoch、128教師Match soft4epoch、128教師Match sharp4epoch。sharp は教師の実Top1とutility上位16の鋭い順位蒸留を半分ずつ組み合わせた損失。**実際の実戦勝敗は学習ターゲットにしていない**。
- 評価指標：Top1完全一致、Top10一致、正規化cross-entropy、公開AQ utility regret。新旧のTop1差に対する80%（2新候補の多重比較補正済み）seed-cluster bootstrap区間を参考表示する。bootstrapは漸近的な近似であり、95%の過去実戦証拠を再解釈しない。
- 性能改善を観測しても**直ちにChampionとしない**。同一holdoutを見て新モデルを選べば、そのモデルで実戦の有意差を主張するには完全未使用の新seedが必要。

## 計算／回帰

- `DeepSetsDraft.forward`の per-mon `phi` をロスターごとに一度だけ計算し、候補集合へのgatherを用いる。旧`phi(pool[codes])`と出力・学習勾配を比較する新たな回帰テストを追加。
- 学習用 `scripts/rnad/train_draft_imitation_128.py` は旧重みのSHA256/shape/schemaを厳密に検証し、新モデルも旧JS推論が受け入れられる`rnad-draft-deepsets-weights-v1`の**別ファイル**に保存する。新モデルのJS/PyTorch実重みParityと実戦強度は別ゲート。
- [研究Workflow](../../../.github/workflows/rnad-i5-draft-imitation-128.yml)は専用manifest更新によるpush時のみ起動し、1writer、実験全体のtrain/heldout上限192Match、CPUモデル等の固定、各ジョブ最大120分/300分。gatherでPIN済みTorchの計算一致テストを**高コストの教師データ収集前**に実行し、全192教師Match保存後にtrainを開始。モデル・メトリクス・個別選択例をArtifactへ保存、Champion昇格とmainマージは実行しない。
- 既存のR-NaD source SHA/engine SHAを追跡し、途中エラーやデータ欠落は「未完了」として扱う。結果を見て途中停止・試合数後付け変更はしない。

**完了条件：** 同一未使用64Matchの模倣数値が存在し、対照との計測上の差を集計した後にのみ「模倣率が上がった／上がっていない」を判定する。これだけでは実戦優位を結論しない。
