# hook・マスクカスタムノード限定移植計画

作成日: 2026-10-06。状態: 限定移植の実装・検証完了。計画作成時の調査記録を保持し、末尾の「実装結果」に今回の結果を追記した。

## 1. 要件と結論

`C:\ComfyUI\custom_nodes\Confyui-krea2-slider-node` の **Krea2 Native LoRA Hooks Fix** と **Krea2 Region Masks** の実装だけを、このリポジトリに追加する。移植元ブランチの学習機能、データセット、プロンプト変更などは取り込まない。

推奨方法は、現在の移植元作業ツリーから対象ファイルを個別に取り出し、このリポジトリの登録処理へ最小限の変更を加える方法。製品コードの追加は Python 4ファイルと JavaScript 3ファイルに限定する。必要なライセンス、専用回帰テスト、検証ツール、既存説明の最小訂正を付随作業とする。サンプルワークフローは今回の範囲に含めない。

**重要: 対象7ファイルはすべて移植元の未追跡ファイルである。** 移植元 HEAD のコミットだけでは実装を取得できない。ブランチ全体の merge、cherry-pick、ディレクトリ全体のコピーは使用しない。

## 2. 調査対象と確定した状態

以下では `SRC/` は移植元、`DST/` はこのリポジトリの相対パスを示す。行番号は調査時点のファイルに対応する。

| 項目 | このリポジトリ（DST） | 移植元（SRC） |
| --- | --- | --- |
| 絶対パス | `C:\Users\inott\Downloads\Confyui-krea2-slider-node` | `C:\ComfyUI\custom_nodes\Confyui-krea2-slider-node` |
| ブランチ | `main` | `feat/anchor-gender-preservation` |
| HEAD | `fe3e111d698a81c6000ba0821634acb9c4f3c39d` | `88ea7f42cf7250bd8ec30833fa28728c33363b5c` |
| 調査開始時の Git 状態 | clean | 追跡済み4ファイルに変更、対象ノード等に未追跡ファイルあり |
| 登録ノード | `Krea2SliderTrainLoRA` の1種類 | 学習2種類と今回の対象2種類 |

根拠（直接確認した事実）:

- `git status --short`、`git branch --show-current`、`git rev-parse HEAD` を両リポジトリで確認した。
- `SRC/nodes.py:18–24` は manifest/selective 関連 import と、対象2ノードの import を別々に持つ。`SRC/nodes.py:235–238` は4ノードを登録する。
- `DST/nodes.py:121–124` は既存学習ノードのみを登録する。`DST/__init__.py:4–6` は V3 エントリポイントを使い、`WEB_DIRECTORY` を持たない。
- `SRC/__init__.py:3` の `WEB_DIRECTORY = './web'` はブランチの既存変更であり、今回の未コミット差分だけを見ても発見できない。
- `SRC/web/` には対象3ファイル以外に `manifest_selection.js` と `selective_manifest.js` がある。Webフォルダー全体のコピーは範囲外コードを混入させる。
- `git diff DST_HEAD --shortstat` を移植元で確認すると、追跡済み差分だけでも 2,184ファイル、244,721行追加、64行削除がある。この集計には未追跡の対象実装を含まない。
- 両者の `pyproject.toml:8` の製品依存関係は同じ。対象 Python 実装の追加依存は既存の `torch` と ComfyUI、標準ライブラリだけである（`SRC/krea2_slider_node/native_hooks.py:6–8,18–19,39–44,94–112,131–133`、`SRC/krea2_slider_node/region_masks.py:6–9`）。

### 調査結果の優先順位と確度

| 優先 | 結論 | 確度 | 根拠 |
| --- | --- | --- | --- |
| 1 | 個別ファイルの抽出で他の学習機能を取り込まずに移植できる | 高（推論） | 2ノードの import は専用の2バックエンドだけを参照し、そのバックエンドには既存学習モジュールへの import がない |
| 2 | マスク編集UIまで再現するには3 JSファイルと `WEB_DIRECTORY` が必要 | 高（事実） | `SRC/web/region_masks.js:1–8`、`SRC/web/region_mask_editor.js:1,7–12,156–162`、`SRC/__init__.py:3` |
| 3 | 検証ツールは現リポジトリ向けに切り出す必要がある | 高（事実） | `SRC/tools/validate_comfy.py:34–53` は selective ノードを要求し、`SRC/tools/validate_native_hooks.py:31–32,38–39` は今回除外するワークフローを読み込む |

「個別抽出で十分」は静的な依存確認に基づく。移植後の実ロード・専用テスト合格は、以下の検証工程で確認する。

## 3. 対象ノードの機能と依存範囲

### hook: Krea2 Native LoRA Hooks Fix

- 入出力は `MODEL → MODEL`。Krea2 の ModelPatcher を clone し、そのインスタンスと後続 clone にだけ互換性修正を設定する（`SRC/nodes_native_hooks.py:7–25`、`SRC/krea2_slider_node/native_hooks.py:110–142`）。
- 修正内容は実パラメータ/バッファの列挙、量子化重みの完全な置換・復元、Hook切替、例外時の復元、`MinVram` によるキャッシュ無効化（同 `:18–107`）。
- LoRA作成、Hook登録・スケジュール、Conditioningへのマスク設定は ComfyUI 標準ノードに任せる。新しい学習ノードや独自Hook生成ノードは追加しない（同 `:1–5`）。
- 非Krea2 MODELと実行中のHookを拒否する（同 `:135–139`）。既存学習ノードへ修正済み MODEL を渡す仕様変更は行わない。

### マスク: Krea2 Region Masks

- 元画像なしで正規化された矩形を2つ編集し、`MASK, MASK, INT, INT` を出力する（`SRC/nodes_region_masks.py:7–35`）。
- 各MASKはCPU上の float32、形状 `(1, height, width)`、矩形内1・外0。初期配置は左右半分。JSON検証、座標clamp、極小矩形の最低1画素を保持する（`SRC/krea2_slider_node/region_masks.py:12–61`）。
- DOM canvasによる移動・四隅のサイズ変更・選択・reset・保存復元・外部JSON入力時の編集停止を含む（`SRC/web/region_mask_editor.js:36–74,124–162`）。
- 性別判定、人物検出、学習用データセットのマスク処理は含まない。領域番号は配置番号である（`SRC/nodes_region_masks.py:14–16`）。

依存関係は次の範囲で閉じる:

```text
既存 comfy_entrypoint → 既存 Krea2SliderExtension
  ├─ 既存 Krea2SliderTrainLoRA（処理・schemaを維持）
  ├─ nodes_native_hooks.py → krea2_slider_node/native_hooks.py → torch / ComfyUI
  └─ nodes_region_masks.py → krea2_slider_node/region_masks.py → torch / 標準JSON

WEB_DIRECTORY='./web'
  └─ region_masks.js → region_mask_editor.js → region_mask_geometry.js
       └─ ComfyUI app.js / addDOMWidget / ブラウザ標準API
```

## 4. ファイル単位の移植範囲

### 必須の新規ファイル

以下だけを同じ相対パスに追加する。未追跡ファイルなので、GitのHEADではなく、確認した作業ツリーファイルから取得する。

| ファイル | 役割・根拠 |
| --- | --- |
| `nodes_native_hooks.py` | hookノードのV3 schemaと実行（SRC `:7–25`） |
| `krea2_slider_node/native_hooks.py` | インスタンス限定のHook互換性修正（SRC `:18–142`） |
| `nodes_region_masks.py` | マスクノードのV3 schemaと実行（SRC `:7–35`） |
| `krea2_slider_node/region_masks.py` | 矩形JSON検証とMASK生成（SRC `:17–61`） |
| `web/region_masks.js` | ComfyUI extension登録（SRC `:1–8`） |
| `web/region_mask_editor.js` | ノード内DOM編集UI（SRC `:7–163`） |
| `web/region_mask_geometry.js` | 正規化・移動・resize・hit判定（SRC `:8–62`） |
| `licenses/D2-MIT.txt` | マスク計算・編集ロジック由来の著作権表示とライセンス。SRC backend `:1–4`、geometry `:1–2` が参照する |

### 既存ファイルへの最小変更

| ファイル | 変更内容 | 持ち込まない部分 |
| --- | --- | --- |
| `nodes.py` | 対象2クラスの import を追加し、`get_node_list()` の戻り値を既存学習ノード＋対象2ノードの3種類にする（DST `:121–124`） | SRCの学習ノード本体、anchor入力、selective/manifest import・ノード |
| `__init__.py` | `WEB_DIRECTORY = './web'` の1設定を追加（SRC `:3`） | エントリポイント方式の変更 |
| `THIRD_PARTY_NOTICES.md` | SRC末尾の D2 由来の段落だけ追加 | 既存通知の置換 |
| `tools/validate_comfy.py` | 3ノードの明示的なID・型確認を追加。既存学習schema・LoRA検証を維持（DST `:27–57`） | SRCのselective/manifest検証 |
| `README.md` | 「カスタムノードは1つ」（DST `:32`）を3種類の説明に訂正し、新ノードの最小接続・検証手順だけ追加 | SRC README全体、追加学習・データセット・サンプル説明 |
| `docs/compatibility.md` | 「One V3 training node」（DST `:11`）に推論用2ノードの追記だけ行う | 学習対応範囲の拡大 |

`tools/validate_comfy.py:41` は現在、YAMLの選択肢を3ファイルと固定しているが、現リポジトリのテスト `tests/test_prompt_files.py:12–20` は既存v2を含む4ファイルを要求する。これは既存検証ツールの不整合であり、登録検証の更新時に現在の4ファイルへ合わせる。YAMLの追加・変更はしない。

### 同梱する専用検証

| ファイル | 方針 |
| --- | --- |
| `tests/test_native_hooks.py` | SRCの専用11テストを移植。量子化、clone、dynamic delegate、復元、例外、標準samplerの領域合成を検証（SRC `:64–231`） |
| `tests/test_region_masks.py` | SRCの既定配置、解像度変更、範囲外矩形、不正JSON/寸法、最低1画素のテストを移植（SRC `:14–62`） |
| `tests/web/test_region_mask_geometry.mjs` | SRCの4テストを移植。serialize、移動、resize、重なり時の選択を検証（SRC `:10–42`） |
| `tools/validate_native_hooks.py` | SRC `main()` のCPU専用テスト起動処理を移植し、`validate_preview`・`validate_workflow_pair` とその呼び出しを除く（SRC `:9–75,78–94`）。11件が実行され、skipされていないことを成功条件にする |
| `tools/validate_region_masks_ui.mjs` | 単独DOM fixture検証を移植。既存PlaywrightとChromeを使い、製品依存に追加しない（SRC `:10–11,44–49,50–112`） |

上記は対象実装の検証だけを目的とする。SRCの他のテスト・toolsはコピーしない。新規ファイルは製品7＋ライセンス1＋検証5の計13ファイル、既存の変更対象は上表の6ファイルとする。

## 5. 明示的に除外する範囲

- `Krea2SelectiveSliderTrainLoRA`、`anchor_strength` とanchor preservationの学習処理（SRC `nodes.py:84–86,97–108,133–232`）。
- SRCの `conditioning.py`、`config.py`、`job.py`、`lora_io.py`、`slider_loss.py`、`training.py` の変更。対象2バックエンドからの依存がないため不要。
- `manifest_files.py`、`native_dataset.py`、`selective_data.py`、`selective_training.py`、selective関連tools・tests。
- `datasets/` 全体、追加・変更された `prompts/`、生成画像、評価結果、学習レポート、`test-results/`。
- `web/manifest_selection.js`、`web/selective_manifest.js`。SRC `nodes.py:18–21` のmanifest学習機能に対応するUIであり、対象マスクUIとは独立している。
- `workflows/` 全体。対象ノード用の6サンプルJSONも「実装部分のみ」という指定に合わせて除外する。現リポジトリにある3つの既存workflowは変更しない。
- SRCの `docs/native-masked-lora-hooks.md`、`docs/region-masks.md` の丸ごと移植。今回除外するサンプルへのリンクと移植元全体の過去テスト成績を含むため、必要な操作説明だけREADMEへ記載する。
- `pyproject.toml` の変更、npm/package構成の追加、ComfyUI本体の編集、インストール済み移植元への書き込み、ブランチ全体の統合。

## 6. 実装手順

1. **状態固定と範囲確認**: DSTのGit状態と既存学習schema、プロンプト4件、既存workflowの内容を記録する。SRC対象ファイルのSHA256を末尾の値と比較し、変化していれば対象ファイルだけ差分を再調査する。ブランチ名やHEAD一致だけをコピーの根拠にしない。
2. **専用検証の導入**: 上記5検証ファイルを追加し、workflow読み込みに依存する部分を除く。ComfyUI専用Pythonから実行する前提を維持し、全skipでも成功になる判定を防ぐ。既存学習ノードのschemaはノードIDで取得し、3種類のID集合を厳密に確認する検証を準備する。
3. **対象実装の追加**: Python 4＋JS 3＋MITライセンスを個別コピーする。専用実装は原則そのまま保持し、selective/anchor/manifest依存を追加しない。著作権ヘッダーを保持してTHIRD_PARTY_NOTICESへD2の1段落を追記する。
4. **登録とWeb配信**: DSTの `nodes.py` へ2つのimportと登録だけ追加し、`__init__.py` へ `WEB_DIRECTORY` を設定する。登録順は既存学習、hook、マスクとする。Webには3ファイルだけを置き、`../../scripts/app.js` の相対参照を維持する。
5. **最小説明・検証整合**: READMEとcompatibilityのノード数説明を更新する。検証ツールで3ノード、4 YAML、MODEL入出力、MASK/MASK/INT/INT、既定左右マスク、Webファイルの存在を確認する。SRCの学習schemaをコピーしない。
6. **検証と範囲監査**: 次節の順で実行し、失敗は対象差分内で修正する。最後に追加ファイルも含めたGit変更一覧を許可リストと比較し、学習コア・prompts・datasets・workflows・他のWeb拡張が含まれないことを確認する。

各工程は独立した確認単位とする。移植元や稼働中のComfyUIを上書きする工程、commit/push、フルモデルでの画像生成は本計画の実装に含めない。

## 7. 受入条件と検証方法

| 条件 | 確認方法 |
| --- | --- |
| 登録ノードが正確に既存学習＋対象2種類 | V3の `get_node_list()` と `GET_SCHEMA()` を実ComfyUIで検証。selectiveノードが登録されていないこともID集合で確認 |
| 既存学習動作・schema・4 YAML・workflowを維持 | 既存pytest一式、移植前schemaとの比較、対象外ファイルのGit差分なしを確認 |
| hook修正は出力patcherに限定され、clone後も維持 | SRC `tests/test_native_hooks.py:92–119` の専用テスト |
| INT8・ConvRot・FP8・非量子化の切替後に重みとscaleを復元 | 同 `:64–89,121–154,172–209` の専用テスト |
| 非Krea2/active Hooksの拒否と、例外復元 | 同 `:156–170,193–209` の専用テスト |
| 標準samplerでマスク領域とdefault領域の予測を合成 | 同 `:212–231` の小テンソルテスト。実画像の完全な画素分離とは区別する |
| MASK出力型・形状・既定左右配置・入力エラーを維持 | schema検証と `tests/test_region_masks.py` |
| UIの保存復元・drag・resize・reset・外部入力時停止が動く | JS単体4テスト、DOM fixture、実ComfyUI画面での確認 |
| Webのロードとノード内DOMが実アプリでも動く | 実サーバーの `/api/object_info`、対象拡張の配信、ブラウザconsole、実workflowの保存再読込を確認 |
| 製品依存と対象外コードが増えない | `pyproject.toml` 不変、追加・変更の許可リスト監査、追加7コードファイルのimport確認 |
| D2の著作権・ライセンスを保持 | backend/geometryヘッダー、MIT本文、THIRD_PARTY_NOTICES追記を確認 |

実装後の基本コマンド（DSTで実行）:

```powershell
python -B -m pytest -q tests/test_region_masks.py
node --test tests/web/test_region_mask_geometry.mjs
python -B -m pytest -q

# <COMFY_ROOT> は comfy/model_patcher.py を含む本体のルート。
& 'C:/ComfyUI/.venv/Scripts/python.exe' -B tools/validate_native_hooks.py <COMFY_ROOT>
& 'C:/ComfyUI/.venv/Scripts/python.exe' -B tools/validate_comfy.py <COMFY_ROOT>

# 既存のPlaywrightパッケージが利用可能な場合。
node tools/validate_region_masks_ui.mjs <既存の@playwright/testの絶対パス>
git diff --check
```

通常Pythonのpytestでは、ComfyUIがimportできないとhook専用11件がskipされる（SRC `tests/test_native_hooks.py:12–21`）。**通常pytestの合格だけでhookの動作確認済みとはしない。** 専用ランナーで11件実行・skip 0件・失敗0件を別に確認する。

`C:/ComfyUI` はこの環境ではデータ/venvの置き場であり、`comfy/model_patcher.py` を含まない。調査時には次の2つの本体が存在したが、稼働中サーバーがどちらを使うかは未確認。実装時は検証先と稼働先を一致させる。

- `C:/Users/inott/ComfyUI-Installs/ComfyUI/ComfyUI`
- `C:/Users/inott/AppData/Local/Programs/ComfyUI/resources/ComfyUI`

静的確認では両本体に `get_key_weight`、`clone`、`get_key_patches`、Hook patch/unpatch、clone callback APIが存在した。存在確認は意味的な互換性の証明ではないため、専用テストを必須にする。実画面検証が実行できない場合は未検証項目として明記し、DOM fixture合格と混同しない。

## 8. リスクと対応

| リスク | 根拠 | 対応 |
| --- | --- | --- |
| SRC作業ツリーの変更により違う実装を移植 | 対象ファイルが未追跡 | SHA256で対象スナップショットを照合し、変化した対象だけ再確認する |
| 学習機能の混入 | SRC `nodes.py:18–21,84–86,133–238`、Webにmanifest拡張あり | 個別コピーと許可リスト監査。共有 `nodes.py` はDSTを編集する |
| サンプル除外で検証ランナーが失敗 | SRC `tools/validate_native_hooks.py:31–39` | workflow依存の検証を切り離し、schema・小テンソルテストで検証する |
| ComfyUI内部API更新によるHook不整合 | SRC `native_hooks.py:18–124` はModelPatcher内部APIを使用 | 検証対象本体を記録し、実APIを使う11件を実行する |
| cloneは内部モデルを共有し、推論中に一時的な重み変更が起きる | SRC `native_hooks.py:48–67,79–89,140`、SRC説明 `docs/native-masked-lora-hooks.md` の共有モデルの制限 | 復元・例外・cleanupテストを維持。同一モデルの並列サンプリング対応を新たに保証しない |
| 通常pytestがhookを実行せず合格 | SRC `tests/test_native_hooks.py:12–21` | 実ComfyUI用ランナーで実行数・skip数を確認する |
| DOM fixtureは実ComfyUIの拡張ロードやVue画面を再現しない | SRC `tools/validate_region_masks_ui.mjs:18–36` はfixture | 実画面でロード、操作、保存再読込を確認する |
| 矩形マスクが人物検出や画素単位の影響隔離と誤解される | SRC `nodes_region_masks.py:14–16`、SRC hook説明のAttention/VAEの制限 | READMEでは配置指定・予測合成として説明し、実画像品質保証はしない |
| lint/typecheck環境がない | 今回の `Get-Command` ではruff/mypyを検出できなかった | 新しい依存は追加せず、利用可能な既存環境を確認。なければAST/JS構文検証・テスト・diff検査で代替し、未実行を記録する |

## 9. 計画作成時の検証証跡と未確認事項

実施済み:

- 両リポジトリのGit状態、ブランチ、HEAD、共有ファイルと未追跡実装を確認。
- Python/JS実装、専用テスト、検証ツール、ノード登録、Web配信、ライセンスの依存を静的に確認。
- このリポジトリで `python -B -m pytest -q` を実行: **33 passed, 1 skipped in 11.84s**、終了コード0。
- 対象ファイルのSHA256を採取。移植元には書き込んでいない。

未実施・未知:

- 対象実装はまだこのリポジトリに追加していないため、移植後のテスト、schema、UIロードは未実施。
- SRC docs記載の過去のテスト合格数は今回の再検証結果として扱わない。
- 稼働中ComfyUIの本体ルート・frontend版、Playwrightの所在、実画面での編集動作は未確定。
- フルサイズKrea2/RAW/TurboでのGPU生成、人物への選択性、境界の品質は未検証。本計画の完了条件に画像品質評価は含めない。
- native調査子エージェントは設定モデルがこのアカウントで未対応のため起動できず、主エージェントが調査した。独立レビューは未実施。

計画作成時の終了条件は、この調査と限定移植計画の作成までだった。その後、ユーザーの実装指示を受け、次節以降に示す範囲で実装した。インストール済み移植元への反映やデプロイは行っていない。

## 10. 移植元ファイルのSHA256

HEADでは識別できない未追跡実装を特定するため、調査時点の生ファイルのハッシュを記録する。コピー前に照合し、改行変換後のDSTとの差を同一性判定に使わない。コピー後は意味的な変更の有無もdiffで確認する。

| SRC相対パス | SHA256 |
| --- | --- |
| `nodes_native_hooks.py` | `2177b604438bb8bb9abc91473ea8079c74301e3252e7221f64fd3187f14ab529` |
| `nodes_region_masks.py` | `f001b0243281dd6454b445fc872e5538806b28e89a218d537f341f6e6e8cae04` |
| `krea2_slider_node/native_hooks.py` | `a7a1f7488fbbfe0b0398a99ec0fe20fc23235c1fb2536a26089e2c4e5ffaf61c` |
| `krea2_slider_node/region_masks.py` | `805d82291c84b914ca8f33c585f07661dee18a7648f7598c6a4b5ef1ba0d3661` |
| `web/region_masks.js` | `0e24c5aa3a0f68833c386fd9d1d408c3e6e49981caeb35e9c87a3fa51052e321` |
| `web/region_mask_editor.js` | `3ca66e892b0b5c8da153fd7a37df944ab2cb1c751da2a1e7a3feaccdfd134234` |
| `web/region_mask_geometry.js` | `5983fa7de2ae7394b1a429e217584cf936681ec72470343fdf9a23c6cdd0e9c6` |
| `licenses/D2-MIT.txt` | `14fb814fb2cbfa5447a2b0740fc34927f1c679dbb48d9268b0b07ae12687e5ae` |
| `tests/test_native_hooks.py` | `db4253b22c89883af8e87abcc8f2cc48528ebe861de70c0ced02622fe53cd997` |
| `tests/test_region_masks.py` | `2abd072a38de30f5186bc4f30346eadde13ae3ab97485d72c512efcb0b36ff58` |
| `tests/web/test_region_mask_geometry.mjs` | `dc3cfc306ff988983e205ea9b8b8df605f3dacc5182e5fea1ce90a51654e9e3c` |
| `tools/validate_native_hooks.py` | `de2dd24a6642afbffffbdf4633b9d8c1854c1f2f70039d9b27599672da1698b7` |
| `tools/validate_region_masks_ui.mjs` | `2a30edbb21c4cef935f1da20a47531b3be4bc72f2d18e4e26ec69d140d9e1e25` |

## 11. 実装結果（2026-10-06）

対象のPython 4ファイル、マスク編集JS 3ファイル、D2 MITライセンス、専用検証5ファイルを追加した。既存6ファイルへの変更は登録、Web配信、検証、説明、著作権表示に限定した。学習クラスのASTとV3スキーマは移植前後で一致し、学習コア、prompts、workflows、製品依存関係は不変である。

### 移植元からの必要な調整

- `tools/validate_native_hooks.py` から対象外workflowの検証を除き、専用11件が未収集・skipでも成功しない判定を追加した。
- `tools/validate_comfy.py` は3ノードをIDで確認し、既存学習ノードの4 YAMLを維持した。
- **実画面で見つかったドラッグ停止を修正した。** ComfyUI frontend 1.53.6のDOMWidgetは `regions.value` の代入時に同期的にcallbackを呼ぶ。移植元のcallbackは `refresh()` → `finishDrag()` を呼ぶため、最初のpointermoveでcaptureを解除していた。`web/region_mask_editor.js` の保存中だけ再入したrefreshを抑え、元のcallbackは引き続き実行する。外部入力・手入力・解像度変更時のrefreshは維持した。
- 同梱DOM検証fixtureにもこの値setterの挙動を反映した。実ComfyUIのマウス操作で修正前の停止と修正後の連続移動を確認した。

### 検証結果

| 検証 | 結果 |
| --- | --- |
| 最終 `python -B -m pytest -q -rs` | **47 passed / 12 skipped**。11件は通常PythonからComfyUIをimportできないため、残り1件は既存の任意ローカル参照がないため |
| ComfyUI用Pythonのnative Hook専用ランナー | **11件合格 / skip 0件**。INT8・ConvRot・FP8・非量子化、clone、dynamic delegate、例外復元、標準samplerの合成を確認 |
| JS幾何テスト | **4件合格** |
| 実ComfyUIのV3検証 | 3ノード、入出力型、左右マスク、既存4 YAML、標準LoRAの強度 -1/0/+1 が合格 |
| 既存学習スキーマ比較 | 入力、出力型・名前、category、output属性が移植前後で一致 |
| 実ブラウザ | ComfyUI **0.37.0**、frontend **1.53.6**、Chrome headless。Canvas/Vue各表示でextension load、DOM binding、drag、resize、save/reload、editor重複なし、解像度変更、外部入力時停止、不正JSON、resetの各10項目が合格 |
| 実APIでのモデル不要の実行 | `Krea2RegionMasks → MaskToImage → SaveImage` が成功。16×8の左右PNGの画素を確認 |
| 静的検証 | Python AST、JS構文、Git diffの空白検査、許可リストによる変更範囲監査を実施 |

検証サーバーはリポジトリ内の一時user/output/temp/cacheとローカル専用ポートを使用した。製品ソースとして新しいworkflowや追加依存を導入していない。テストログ・schema比較・画面・API結果は、Git除外対象の `test-results/hook-mask-port/` に保存している。

### 検証の限界とレビュー

- 既存Playwrightパッケージが見つからなかったため、同梱のPlaywright DOM検証コマンド自体は未実行。代わりに依存不要の一時Chrome CDPプローブで、実ComfyUIの両表示を検証した。製品にCDPプローブは同梱しない。
- ruff/mypyは利用可能な環境になく、専用lint/typecheckは未実行。構文解析・テスト・差分監査で代替した。
- installed roleモデルがこのアカウントで未対応のため、独立子エージェントのレビューは利用できなかった。実装後に別工程として自己レビューを行い、範囲、ライセンス、保存復元、callback再入、skipによる偽合格を点検した。
- フルサイズKrea2のGPU画像生成・人物選択性・境界画質は今回の範囲外。稼働中のインストール済み移植元は上書きしていない。実装・検証の段階ではcommit/pushを行わず、その後のユーザー指示に基づいて別途行う。
