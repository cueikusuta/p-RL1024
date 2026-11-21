# I5｜2026-11-17 夜間10時間・20並列のR-NaD先行リーグ探索（ユーザー明示変更）

## 判断と制約
ユーザーは2026-11-17夜に、ドラフト・選出・戦闘の教師Top1模倣ゲートがまだ揃っていないことを承知で、**直ちにリーグ学習を開始すること**、手法・対戦相手はAIが選び、GitHub-hosted runnerを可能な限り使用し、**約10時間規模**で進めるよう明示指示した。以前の「三相Top1を満たすまでリーグ禁止」は**正式Champion昇格・本番採用向けゲート**として維持し、これとは別に**隔離された探索リーグ学習**を開始する。リーグ学習そのものはNeuRD/V-traceを用いる実体のある更新で、偽の開始フラグや模倣学習の言い換えではない。

未修正P1：`scripts/rnad/league_entry_gate.py` は自己申告JSONからリーグ許可を返せるとの独立AIレビュー #5938960571がある。この探索では同ゲートを**呼ばず**、自動Champion昇格・PRマージ・本番配置・強度確証宣言を一切行わない。この開始は旧ゲートを達成したものとして記録しない。P1修正と再独立レビューは別途必須。

## 実行法・10時間計画
- 専用Workflow `.github/workflows/rnad-i5-night-league-10h.yml`を、唯一の事前登録Launch JSON追加pushで起動。学習・検証主体はGitHub-hosted `ubuntu-latest`、セルフホストランナーは使用しない。
- Stage0/NeuRD/V-trace/権威完全4勝Match/合法公開情報/受理後trajectory/チェックポイント保存・完全状態復帰の既存実装 `scripts/rnad/train.py`を利用。ハイパーパラメータ `eta=0.2`、magnet更新4Match周期、Adam、value/target/magnet/RNGとoptimizer/schedulerを**全て保存**。
- 20個の独立single-writer系列で`matrix.shard:0..19`、`max-parallel:20`。GitHubの実際の実行枠が20未満のときはキューで待機するため、10時間は**計画値であり保証ではない**。
- 時間予算は**1系列あたり2波×4時間48分＝9時間36分の学習処理**、セットアップ・Artifact転送・検証込みで約10時間を目標にする。Github-hostedジョブ6時間上限より十分短い、各波`timeout-minutes=345`。段階2は波1の不変チェックポイント/初期ファイルSHAと同じGit commitから再開。いずれもGitHub-hosted runner。
- 学習seedは各系列`2600001+50000*shard`から独立。モデル元・プール・feature/学習元SHAをバイト検証。20系列は共通のoptimizerに同時書込をしない。別のシリーズの学習重みを同期集約・平均しない。
- **15系列**：既存実証済み**v3**権威Matchリーグ実装（`--features-v3 --switch-guard anti-loop-v1 --league-pool`）。過去の学習済みv2 `base128/base256/c005`を決定的にv3へ移植・SHA固定し、全3種を凍結対戦相手とする。学習側の対戦席をモデルローテーションとともに交互に切替。対戦相手は随時学習更新しないので、論文の人口型PFSPやElo適応リーグではなく**凍結アーカイブリーグ**。
- **5系列**：最新の研究用**v5** `battle-focus256`重み（Recovery Run #44037414890）をSHA検証してwarm-startし、本人対本人で全相`--features-v5`のR-NaD/NeuRD/V-trace自己対戦を行う。これは選出A4096 `core20+lead6`重みを直接取り込んだ統合モデルではない（互換モデルではないため）。
- 半数系列`updates_per_match=1`、半数`4`（同一受理Matchへの4反復時はoff-policy importance ratio補正）として探索幅を確保。固定エンジンsourceと全モデル重みSHA・各試合seed/Match-ID・更新数を検証する。
- 1Match完了ごとに学習済model・価値ネット・磁石・Adam・LR scheduler・RNG・受理MatchIDをimmutable checkpointで保存、旧checkpointは最新1件を保全して削除。各波末に20 Artifact、2波終了時はSHA・全MatchID/seed重複を集約して報告。中断系列は成功した直近Artifactのみ救済し、偽の完走扱いをしない。
- 今回、時間だけを費やしたことを「強化成功」「Championになった」と判定しない。性能判断には独立大量完全4勝Match、両席対照、95%seed-cluster CIを別途必須とする。今夜の成果物は未昇格の研究用20系列のチェックポイントと完全受理ログ参照である。

## 既存の模倣研究との並行と優先事項
- [Run #44096608535](https://github.com/cueikusuta/p-RL1024/actions/runs/44096608535) はDraft/Battle 256→512完全教師Match学習・新規512Matchの独立比較を**既に開始しており**、今夜のリーグはその実行内容・データを重複生成しない。GH runnerに枠競合があればキューへ。
- Draft/Battleが依然教師模倣目標未達なら結果を後日分析し、学習規模以外のfeatures/teacher signalを検討する。ユーザー最新指示で研究リーグ開始を許可されたことと、同一統合方策の Draft>50%/Selection>=50%/Battle>=80% と補正下限の正式研究ゲートが未達であることを明確に区別する。
- 今夜は**自動Champion選出やmainへのマージを実装しない**。次回は独立大量評価で各候補の強さを測り、P1修正・独立再レビューと統合方策の確立を進める。

## 成果物
- source: `scripts/rnad/night_league_2026.py`
- regression: `test/rnad_night_league_2026_test.py`
- Actions: `.github/workflows/rnad-i5-night-league-10h.yml`
- one-shot trigger: `docs/research/rnad/OVERNIGHT_LEAGUE_LAUNCH_2026-11-17.json`
- wave1/wave2: `rnad-i5-overnight-v3-wave{1,2}-<shard>` （命名は歴史的互換でv5自己対戦も含む）
- aggregate: `rnad-i5-overnight-10h-complete-series-verification`
