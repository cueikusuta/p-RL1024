# I5｜ドラフト・戦闘追加学習後の独立教師Top1評価（再実行なし）

## 検証対象・既存成果の再利用
- ユーザー2026-11-18の指示「今評価してください」を実施する。既存[Run #44096608535](https://github.com/cueikusuta/p-RL1024/actions/runs/44096608535)では**学習済みモデルの保存まで成功**。教師データ生成 `gather_fresh` は seed `2522144` で `Accepted public AQ teacher does not maximize its declared utility` を発生し、独立評価は未実行。**新たな追加学習は行わない**。新旧すべて事前凍結の重みをSHA・モデル台帳で検証する。
- Draft旧：`soft256-e6.json`、新：`draft512-e6.json`。Battle旧：battle-focus256 v5、新：battle-focus256重みから新規256受理Matchで6epochs継続した512研究モデル。教師・特徴量ソースは全て `022aa9cd5b4ac37514df1d845ea00c11c3c13a5c` に固定。
- 旧モデルの以前の個別研究値 Draft 41.60%／Battle 75.84%は古い128Match上でモデル選択にも使われているため、**新規評価での旧モデル再測定値と混同しない**。

## 教師エラーの扱い：品質ゲートは一切緩めない
- seed `2522144` は公開AQ教師の採択行動が、その場で算出した公開効用スコアの最大値と一致しなかった。**原因は別途調査対象**。オリジナルの `aq-public-utilities.mjs`、teacher policy、権威ゲーム、`export-teacher.mjs`、完全4勝判定、`draft_review.teacher_rank==1` を変更せず、**厳格検証に通ったTraceしか保存しない**。
- 既知の効用不一致は**教師Match丸ごと不採用**（受理済みの成功試合と決して数えない）。各seed・除外理由をManifestに記録し、効用不一致**以外**のエラー、モデル不整合、Source SHA・完全4勝・Trace SHA不一致はすべて即失敗。
- 事前固定の4独立seed列：`2522001+4×試行番号+shard` （shard=0,1,2,3）。各系列は先頭から順に、チェックを通過した**最初の128完全Match**を採用。最大256試行/系列。4系列で**512完全Match**を確保。データ選択は**モデル推論を呼ぶ前**に確定し、旧/新モデルの出力で除外や閾値調整を行わない。seed `2522144` はshard 3の36試行目に含まれ、実際に除外されたことを集約時の契約で確認する。
- ただし、これは「任意の全AQ教師Match」の母集団ではなく、**公開効用の内部整合性を通過した完全Matchという条件付き母集団**。改善有意性があっても全Matchへの無条件な外挿はしない。除外割合・seed・理由を全て公開する。**失敗seedを無断で正常扱いしたり、studentの結果によって選択しない**。

## 判定
- 同一512独立完全Match seedで、旧/新Draft・旧/新Battleをそれぞれ対応付けて評価。Top1は教師の**実際に権威受理されたアクションの一致率**。Battleの時系列memoryは席別に保持。試合内の多数の判断を独立標本として過大計算しない。
- 独立seed cluster単位の平均差を10,000反復bootstrapし、両側近似95%信頼区間を計算。**改善幅CI下限>0**なら改善を支持、ゼロを含めば未確証。別に3相Bonferroni相当familywise80%片側Hoeffding下限を参考値として報告（Draft>50%、Battle>=80%）。今回の個別学生を集めて「同一統合3相policyゲート合格」とはしない。
- 新旧重みのバイトSHAと全512教師Trace SHA、モデルとは独立に確定したseed/除外一覧、評価レポートはGitHub Actions Artifactで90日保存。独立ChatGPTレビューP1 `league_entry_gate.py`の自己申告証拠問題は未修正のため、**正式リーグゲート入力・Champion昇格・PRマージには用いない**。別のユーザー認可による夜間R-NaD探索リーグとは独立。
- 実行はGitHub-hosted `ubuntu-latest`の4並列Trace収集＋1評価ジョブ（保存済みモデル再利用）。`scripts/rnad/collect_strict_teacher_512.py`, `merge_strict_teacher_512.py`, `draft_battle_512_qualified_eval.py`、`.github/workflows/rnad-i5-draft-battle-512-qualified-confirm.yml`。一回のLaunch JSON追加以外では再起動しない。
