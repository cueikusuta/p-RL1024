# カタログの出典と利用条件の調査

調査日: 2026-11-14。対象: p-RL1024 commit `9924a9f3e28bd90800ee297a242b447660cd7c70` の `data/catalog/`。この文書は利用許諾や法的適合性を認定するものではない。

## 2026-11-14 の続行判断と残る停止条件

所有者は、Pokémon Showdown や PokeAPI の公開事例を踏まえ、非商用のファンメイド実験としてデータと自作実験コードで続行すると決定した。この決定は、権利者の許諾取得や利用条件の確認を意味しない。画像・音声などの素材、遊べるゲーム本体は含めない。個別の文章・表現の転載が具体的に見つかった場合、そのデータの利用拡大を止め、削除または置換を先に行う。新しい出所を追加する前には出所と条件を記録する。

以下は、この判断以前の調査で確認した根拠であり、未確認事項として引き続き保持する。

- Pokémon Support は、キャラクター・名称などの Pokémon 知的財産をプロジェクトに使用または関連付けしないよう求めている。[公式サポート](https://support.pokemon.com/hc/en-us/articles/360000634094-Can-I-use-Pok%C3%A9mon-images-or-materials)
- Game8 の利用規約第7条は、サービスコンテンツについて、サービスが予定する態様を超える複製や公衆送信などを制限している。[Game8 利用規約](https://game8.jp/terms)
- `production-sets.json` は Game8 の記事を `build-source` 等として記録する。URL があることだけでは、その記事からどの表現を複製したか、規約が各値にどう適用されるか、許可の有無は判定できない。

上記の適用範囲は未確認であり、違反は確定していない。許諾・独立に作成したデータ・利用条件に適合する代替の確認は未実施。2026-11-14 の所有者判断に従い、個別の転載が具体的に判明した場合の停止条件を適用する。

## 再開前の手当て

- Showdown の参照 commit `b8f23c99629132611ba59e99480cb148ea8c5d3d` の MIT 本文・著作権表示を [LICENSES/pokemon-showdown-MIT.txt](../LICENSES/pokemon-showdown-MIT.txt) に保存し、README から参照する。対象にはカタログデータの参照に加え、同 commit を機構の参照として記載する `src/battle.js` と `src/champions-stats.js` を含む。コード全体の由来調査や第三者の Pokémon 権利の確認は未完了。
- `production-sets.json` の各709件には構造化データのほか `strategy`、`costRationale` などの文章がある。Game8、GameWith、Altema、Pokémon Zone の各1件について、登録済みの短い `strategy` 文字列を参照先ページの取得テキストと照合し、完全一致は見つからなかった。これは4件の限定的な文字列検索であり、過去のページ版、残りの文章、翻案・類似表現、Smogon、Bulbapedia を含めた全文照合は未実施。具体的な転載を確認した箇所はこの範囲ではないため、文章の削除や書き換えはしていない。[Game8 の例](https://game8.jp/pokemon-champions/812736)、[GameWith の例](https://gamewith.jp/pokemon-champions/575742)、[Altema の例](https://altema.jp/pokemonchampions/shandera-ikuseiron)、[Pokémon Zone の例](https://www.pokemon-zone.com/champions/ranked-seasons/singles/)。
- Private 本体 URL 70件は `research.source.url` 69件、`research.priorProposal.source.url` 1件に存在する。実験コードでこれらのプロパティを直接参照する箇所は検索範囲で見つからなかったが、カタログ全体は実行時に読み込まれ、学習・評価の source fingerprint と checkpoint の互換性に含まれる。実行への影響がありうるため、指定どおり URL を変更せず残す。除去後の digest 再検証も行っていない。
- AQ-Base 対戦の分離 snapshot は本体研究 commit `300b4450969b65c146fe1c9e911d3e9bc9f615dd` の対戦コードを参照する。含まれる6件の catalog ファイルは、上記の既存 `data/catalog/` と SHA-256 がすべて一致し、新たなカタログの出所は追加していない。対戦コード自体は本体研究ブランチの自作コードで、各ファイルの元 Git blob と SHA-256 を `provenance/manifest.json` に記録する。

## ファイル単位

| ファイル | 確認できた出所手掛かり | 利用条件と未確認事項 |
|---|---|---|
| `battle-data.json` | `referenceCommit` に Pokémon Showdown の commit `b8f23c99629132611ba59e99480cb148ea8c5d3d` を記録。URL はなし | [同 commit の LICENSE](https://github.com/smogon/pokemon-showdown/blob/b8f23c99629132611ba59e99480cb148ea8c5d3d/LICENSE) は MIT の表示を要求する。各データと第三者の Pokémon 権利への適用は未確定。移植元・変換過程・表示の十分性を要確認 |
| `production-sets.json` | `sources`、`research` 等に下表の URL を記録 | URL は出所の手掛かりでありライセンスや転載許可ではない。元の文章・構成・データ値の対応、権利、許諾を未確認 |
| `cpu-special-sets.json` | Smogon の `setReference` 24件と Showdown の `referenceCommit` | 参照先の記載と実データの関係、原著作者と個別の許可は未確認 |
| `cpu-legendary-additions.json` | ファイル内に URL はなし | 参照元・作成過程・権利者・許諾が未確認 |
| `labels-ja.js` | ファイル内に URL はなし | 日本語表記の出所、権利と許諾が未確認 |
| `runtime-version.json` | engine 互換性と version | 外部データの来歴を示す情報は含まれない。値の生成過程は未確認 |

## `production-sets.json` の参照先

URL を持つ文字列を再帰的に列挙した件数。重複参照を含み、利用件数・複製件数ではない。`github.com` はパス先頭の所有者／リポジトリでも区分した。

| 出所 | URL 件数 | 確認した条件または未確認の理由 |
|---|---:|---|
| Pokémon 公式のニュース・roster 2ホスト | 2 | [Pokémon Support](https://support.pokemon.com/hc/en-us/articles/360000634094-Can-I-use-Pok%C3%A9mon-images-or-materials) の使用に関する要請を確認。各データの具体的許諾は未確認 |
| `github.com/smogon/pokemon-showdown` | 966 | [参照 commit の MIT LICENSE](https://github.com/smogon/pokemon-showdown/blob/b8f23c99629132611ba59e99480cb148ea8c5d3d/LICENSE) を確認。参照された個別ファイルと実際の転用範囲は未確認。別途 Pokémon の権利もある |
| `github.com/cueikusuta/pokemon-battle-factory` | 70 | 本体への参照。個々の参照先とそれに紐づく外部データの権利は未調査。公開先からの参照の妥当性も未確認 |
| `gamewith.jp` | 516 | [利用規約](https://gamewith.jp/terms) の権利帰属・禁止事項を確認。個別記事と実データの関係、許諾の適用は未確認 |
| `game8.jp` | 486 | [利用規約第7条](https://game8.jp/terms) の利用制限を確認。個別記事と実データの関係・許諾の適用は未確認 |
| `www.pokemon-zone.com` | 58 | [利用規約](https://www.pokemon-zone.com/terms/) を確認。個別の build データや Pokémon 知的財産を再配布できることは未確認 |
| `altema.jp` | 15 | [利用規約](https://altema.jp/rule) を確認したが、本データの再利用許可と個別記事の権利は未確認 |
| `dex.pokemonshowdown.com` | 14 | Showdown の表示サイトへの参照。サイト掲載物と repository LICENSE の対応・個別権利は未確認 |
| `www.smogon.com` | 9 | 個別の Smogon 掲載物のライセンス・許諾が未確認。Showdown server の MIT をそのまま適用しない |
| `pokemondb.net` | 6 | 個別の利用条件、対象データとの関係、許諾を未調査 |
| `bulbapedia.bulbagarden.net` | 4 | [著作権方針](https://bulbapedia.bulbagarden.net/wiki/Bulbapedia:Copyrights) は原則 CC BY-NC-SA と例外を記載。参照部分が対象か、条件充足と Pokémon の権利は未確認 |
| `game8.co` | 4 | 日本語サイトと異なるドメイン。適用規約と個別資料の許可は未調査 |
| `rotomlabs.net` | 2 | 個別の利用条件と転載の有無は未調査 |
| `oraswiki.com` | 2 | 個別の利用条件と転載の有無は未調査 |
| `serebii.net` / `www.serebii.net` | 各1 | 個別の利用条件と転載の有無は未調査 |
| `www.destructoid.com` | 1 | 個別の利用条件と転載の有無は未調査 |
| `www.theclick.gg` | 1 | 個別の利用条件と転載の有無は未調査 |
| `strategywiki.org` | 1 | 個別ページのライセンスと転載の有無は未調査 |

`cpu-special-sets.json` の Smogon 24件は上表に含めていない。PokeAPI は今回の対象ファイルの URL としては確認していない。URL がないことは参照していない証明ではない。

## 未完了の確認

1. 各参照がリンクのみ、事実の照合、文章・構成・データの複製のどれかを個別にたどり、生成履歴と照合する。
2. 権利者・利用規約・個別許諾・必要な表示を確認し、継続公開や利用拡大の可否を所有者が判断する。法的な結論が必要な場合は適切な専門家に確認する。
3. 利用できない、または確認できないデータは、許諾済みの代替にするか、公開・実験の範囲を見直す。データが変わる場合はスナップショットと実験結果の再現性を再検証する。

## 数え方

`production-sets.json` を JSON として読み、全ての dict と配列を再帰して `http://` または `https://` で始まる文字列の URL host を数えた。サイトの実ページへの一括アクセスや scraping は行っていない。GitHub host のうち上記以外の所有者は対象データに現れなかった。
