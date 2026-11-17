# I5｜A4096選出の3フェーズ実行基盤への統合：実戦合法性smoke

## 目的・範囲
現在の選出A4096単体は新規未使用教師2048完全MatchでTop1 **55.26%**、3相Bonferroni相当selection-only80%片側下限 **52.71%**を達成した（[Run #44067733118](https://github.com/cueikusuta/p-RL1024/actions/runs/44067733118)）。A2048との直接実戦選出-onlyは512独立seed×両席＝1024完全4勝Matchで勝率 **53.61%**、seed-cluster近似95%CI **[50.98%,56.25%]**（[Run #44085352444](https://github.com/cueikusuta/p-RL1024/actions/runs/44085352444)）。

次の統合では、ゲームの**権威受理後にのみ公開される**3フェーズ間の情報（自軍確定6体→選出確定3体→戦闘）を、凍結モデルの実動作で検証する。これは**モデル統合の受理smoke**であって、学習済み3相R-NaD方策の達成・同一教師Top1成功の証拠ではない。

## 凍結モデル・評価条件
- ドラフト：2026-11-17のRun #44032488678（教師source `022aa9cd5b4ac37514df1d845ea00c11c3c13a5c`）のDeep Sets `soft256-e6.json`、研究台帳のSHA256でバイト検証。過去未使用128教師MatchのDraft Top1 **41.60%**、目標 **>50%** に未達、かつ多重比較補正下限の達成なし。
- 選出：Run #44085352444で数値一致確認済みのA4096 JS actor、学習元教師source `a7224b05eeef49650794b24546576119dffed951`、事前固定モデル `selection-A-4096-e60-s90252.pt`。公開情報/自軍情報のみと権威合法60候補を使用。
- 戦闘：`createBuiltinPhaseAgents({battle:'basic'})`（学習済みbattle-focusの重みでは**ない**）。別研究のv5 battle-focusモデルは独立128MatchでTop1 **75.84%**、目標 **>=80%** に未達。
- 実装：`scripts/rnad/smoke-a4096-threephase-composite.mjs`で3つのseat-scoped phase agentsを`runPhaseMatch`に接続。ドラフト受理を検証してから選出、選出受理を検証してから戦闘へ移行。権威受理前hand-offは禁止。
- 固定seed `2501001,2501002`の**2完全4勝Matchは契約smoke限定**。少数試合で勝率や強度の判定を絶対にしない。同じモデルを両席に配置し、同一モデルの2局の勝敗は統計実験にならない。
- 別ソース・別学生の合成を同一統合policyの教師Top1確認データと**扱わない**。同じsource・teacher・authorityから未使用Matchで実測した3相の観測を同一policy ID／run ID／manifest SHA／同一heldout seed集合で束ねる必要がある。
- [リーグゲート](./LEAGUE_ENTRY_GATE_2026-11-17.json)は Draft >50%・Selection >=50%・Battle >=80%かつ3相Bonferroni補正familywise80%片側下限を要求。別ChatGPT Chat独立レビューも必要。

## 後続作業と安全ゲート
1. 受理smokeでsource/model SHA・合法action・先発と6体/3体の引き継ぎを検証。失敗した場合は原因修正、すでに成功した教師学習・実戦512seed比較の無制限再実行はしない。
2. **Draft・Battle**のモデルを公開情報・合法性契約で同一評価条件に合わせて教師模倣改善する。Draft 41.60%およびBattle 75.84%は過去の異なる研究方策の値で、A4096選出と同一policyの評価値としては使用不可。
3. 同一統合政策を未使用の十分な独立教師4勝Matchで3相同時評価。**統計ゲートが全て成立するまで** R-NaDリーグ学習禁止。
4. Champion昇格は、さらに実戦対元AQ-Base独立両席paired95%CI、CI、別ChatGPT Chat独立レビューの完了が必要。PR #6はDraft、main未マージ。
