# I5 ドラフト改善1〜4：実走の比較結果（2026-11-17）

対象: [Issue #5](https://github.com/cueikusuta/p-RL1024/issues/5)／[Draft PR #6](https://github.com/cueikusuta/p-RL1024/pull/6)。旧R-NaDモデルとcheckpointは保全、研究用の独立ドラフトモデルはゲーム実行経路に未接続。

## 実走・出典

- [Run #43984108897](https://github.com/cueikusuta/p-RL1024/actions/runs/43984108897): **success**。GitHub-hosted Ubuntu CPUのみ。学習源コミット 1f57fa14197a2aa25e127d8409839bf1c46af491、frozen engine ea81165993110ee9fb3a09dcb97c6708c0d7397b。
- 実際に権威Matchで受理された公開AQ戦略教師の16完全Match / 174ドラフト判断で学習。train seed 2442001..2442016。heldoutは別4完全Match / 50ドラフト判断、seed 2443001..2443004。各モデル2 epoch。
- Deep Setsのモデル・アーキテクチャ・ハードラベル vs 順位蒸留・教師データ4対16Matchの比較を同一holdoutで実施。保存した実験用重みJSON/個別model SHA/出力report JSON/具体的なピック比較MarkdownはRun Artifactに30日間保存。

| 条件 | 学習ドラフト判断 | 未使用Top1 | 未使用Top10 | 正規化CE（低いほど良い） | 正規化AQ公開utility regret（低いほど良い） |
|---|---:|---:|---:|---:|---:|
| 加算型＋hard Top1＋16 Match | 174 | 2% | 26% | 0.981 | 0.210 |
| Deep Sets＋hard Top1＋16 Match | 174 | 10% | 40% | 0.627 | 0.146 |
| Deep Sets＋soft utility順位＋4 Match | 42 | 4% | 26% | 0.800 | 0.205 |
| **Deep Sets＋soft utility順位＋16 Match** | **174** | **14%** | **50%** | **0.589** | **0.107** |

今回4条件ではDeep Sets＋soft順位＋16Matchが全指標で最多/最小。ただしこれは**教師到達局面における模倣結果**であり、対戦勝率の向上ではない。50判断の単一holdoutブロックであり、統計的強度確証はない。旧v4 Top1 6.52%・Top10 19.02%とはデータ・学習・評価条件が異なり、直接の改善率比較は不可。

## 実例

同一未使用 seed 2443001・席0、570合法候補の局面：
- 教師が実際に受理された選択：デスバーン／エレザード／カモネギ／ジュナイパー（ヒスイのすがた）／メガプテラ／メガカラマネロ。
- 加算型 hard Top1 モデルの予測：メタモン／デスバーン／カモネギ／ジュナイパー（ヒスイのすがた）／ガラルマッギョ／メガブリガロン。教師utility regret 0.733。
- Deep Sets hard Top1 モデルの予測：教師と完全一致。regret 0.000。

この学生の選択は、**教師が到達した局面での予測**であり、学生自身の4勝Matchの受理行動ではない。ユーザー向け閲覧では教師との一致率だけでなく、選択6体・予算・各候補のutility順位・regretも見る。

## 現在の限界と次の強度ゲート

1. 深いモデルは個体の自席能力値・タイプ・役割タグ・費用・carryを集約するpermutation-invariant研究構造。相手非公開情報は入力せず、受理済み教師データ以外を学習しない。
2. Deep Setsの研究重みは元のGRU/V-trace/R-NaD policyではない。実戦投入にはversioned policy adapter/ONNX又は同等推論、合法accept/公開情報一致、記憶・受理callback検証が必要。
3. 初回は単一train/heldout seed系列。別holdout・教師data増量再現、教師以外の局面のclosed-loop検証、元AQに対する完全4勝Match・両席paired Δ/95%CI・旧v3比較なしではChampion昇格しない。
4. PR #6はDraft、main未マージ。最新HEAD必要CI・別ChatGPT Chatの独立レビューを行ってから採用判断する。小標本Top1=14%を最終AIの強さとは呼ばない。

具体的な学習操作と記録形式は [ドラフトピックレビュー案内](./DRAFT_PICK_REVIEW_2026-11-17.md) に記載。
